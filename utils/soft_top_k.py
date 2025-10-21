import time
import numpy as np
import torch


class SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)

        return torch.where(e>0,t+torch.log(z)-torch.log(ab),s-torch.log(z)+torch.log(ab))
 
    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        # Sprawdzenie wymiarów
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"
        
        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]  
            scaled.div_(alpha)   # Dzielenie in-place

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            # wyliczamy funkcje 2 * L
            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)
            
            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)
            
            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)
            
            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0), 
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i
            )
            return b

        b = finding_b()

        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)
        
        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)
        
        torch.where(sign_x, 1 - p, p, out=p)
        
        # Zapisanie dla backward pass
        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    
    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha

        x.abs_().neg_()
        q = torch.softmax(x, dim=1)

        torch.mul(q, grad_output, out=x)
        grad_k = x.sum(dim=1, keepdim=True)

        grad_r = grad_k - grad_output
        grad_r.mul_(q).mul_(S)

        q.mul_(r)
        x.mul_(S / alpha)  # grad_alpha = (S / alpha) * x
        r.sub_(q.sum(dim=1, keepdim=True))
        x.mul_(r)  # grad_alpha.mul_(r)
        grad_alpha = x.sum()  # grad_alpha = grad_alpha.sum()
        return grad_r, grad_k.squeeze(1), grad_alpha, None


def soft_top_k(r, k, alpha, descending=False):
    return SoftTopK.apply(r, k, alpha, descending)
    
    
def check_value(x, v, text):
    assert x.shape == v.shape, f"Shape mismatch: {x.shape} vs {v.shape}"
    def fun():
        if isinstance(x, torch.Tensor):
            return torch.allclose, torch.linalg.norm
        else:
            return np.allclose, np.linalg.norm

    function, dist = fun()
    check = None
    for tol_exp in range(-15, 0):
        if function(x, v, rtol=1e-05, atol=10 ** tol_exp):
            check = f"Error within atol=1e{tol_exp}"
            break
    if check:
        print(f"✅ - {text} ({check})")
    else:
        print(f"❌ - {text} [dist: {dist(x - v):.4f}]")
        print(f"Expected: {v}")
        print(f"Got: {x}")


def print_time_stats(times, name):
    if not times:
        return
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\n{name} time stats (seconds):")
    print(f"\033[0;1;35m  Average: {avg:.4f}\033[0m")
    print(f"  Min:     {min_t:.4f}")
    print(f"  Max:     {max_t:.4f}")
    print(f"  All times: {[f'{t:.4f}' for t in times]}")
    
    
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # ==============  Parameters  =================
    use_gpu = False
    use_gpu = True

    descending = False
    # descending = True
        
    h = 1e-5

    bs = 10
    n = 10**6
    # =============================================

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"{device=}\n")

    factory_kwargs = {
        "device": device, 
        "requires_grad": True
    }

    forward_times = []
    backward_times = []
    for i in range(10):
        # bs = np.random.randint(1, high=100, size=None)  # zakomentuj jak chcesz te same wymiary tensora
        # n = np.random.randint(5, high=10_000, size=None)  # zakomentuj jak chcesz te same wymiary tensora

        alpha = torch.tensor(np.random.rand(), **factory_kwargs)
        
        r = torch.randn(bs, n, **factory_kwargs)
        k = torch.tensor(np.random.rand(bs) * n, **{**factory_kwargs, "dtype": r.dtype})
        
        print(f"bs={bs}, n={n}, alpha={alpha.item()}")
        assert alpha.dtype == k.dtype == r.dtype, f"You have different types of tensors: {alpha.dtype=}, {k.dtype=}, {r.dtype=}"
        
        # For Backward computation
        v = torch.randn_like(r)

        # Forward pass
        start_forward = time.perf_counter()
        prob = soft_top_k(r, k, alpha, descending)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        forward_time = time.perf_counter() - start_forward
        forward_times.append(forward_time)
        print(f"\033[0;32mForward pass time: {forward_time:.4g} s\033[0m")

        # Test sum
        test_sum = prob.sum(dim=-1)
        check_value(test_sum, k, "test sum")
        
        # ======================================================
        print("=" * 10, "Gradients", "=" * 10, sep="   ")
        
        # Backward pass
        start_backward = time.perf_counter()
        r.grad = None  # Clear gradients
        k.grad = None
        alpha.grad = None
        prob.backward(v)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        backward_time = time.perf_counter() - start_backward
        backward_times.append(backward_time)
        print(f"\033[0;34mBackward pass time: {backward_time:.4g} s\033[0m")
        print(f"\033[0;33mTotal time: {forward_time + backward_time:.4g} s\033[0m")
        
        numerical_derivative = (soft_top_k(r + h * v, k, alpha, descending) - soft_top_k(r - h * v, k, alpha, descending)) / (2 * h)
        check_value(r.grad, numerical_derivative, "grad r")
        
        numerical_k_grad = (torch.mul(v, soft_top_k(r, k + h, alpha, descending) - soft_top_k(r, k - h, alpha, descending)) / (2 * h)).sum(1)
        check_value(k.grad, numerical_k_grad, "grad k")
        
        numerical_alpha_grad = torch.mul(v, soft_top_k(r, k, alpha + h, descending) - soft_top_k(r, k, alpha - h, descending)) / (2 * h)
        check_value(alpha.grad, numerical_alpha_grad.sum(), "grad alpha")
        print("\n")

    print("\n\n", "=" * 75, sep="", end="")
    print_time_stats(forward_times, "Forward pass")
    print_time_stats(backward_times, "Backward pass")
