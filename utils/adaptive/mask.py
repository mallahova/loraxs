from typing import Optional

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt


class WeightMaskingLinear(torch.nn.Linear):
    """
    A Linear layer with dynamic masking applied to its weights.
    """

    def __init__(
        self, in_features: int, out_features: int, rank_min: int, rank_max: int, tau: int = 0, bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("ranks", torch.arange(rank_min, rank_max + 1))
        self.rank_min: int = rank_min
        self.rank_max: int = rank_max
        self.tau: int = tau
        self.s: Optional[torch.Tensor] = None
        self.alpha: Optional[float] = None
        self.index: Optional[int] = None

    def log_matrix_to_wandb(self, matrix: np.ndarray) -> None:
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="gray_r", interpolation="none", vmin=0, vmax=matrix.max())
        plt.colorbar(label="Value")

        num_rows, num_cols = matrix.shape
        plt.xticks(ticks=range(num_cols), labels=range(num_cols))
        plt.yticks(ticks=range(num_rows), labels=range(num_rows))

        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.title(f"Rank: {self.s:.2f}, alpha: {self.alpha:.2f}")

        if wandb.run is not None:
            wandb.log({f"mask_{self.index}": wandb.Image(fig)})
        plt.close(fig)

    def _create_mask(self, p: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(self.weight)
        for i in range(self.rank_min, self.rank_max + 1):
            tensor = torch.zeros_like(self.weight, device=self.weight.device, requires_grad=False)
            tensor[:i, :i] = 1
            mask += p[i - self.rank_min] * tensor
        return mask

    def _create_discrete_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.weight)
        discrete_rank = int(torch.round(self.s).item())
        mask[:discrete_rank, :discrete_rank] = 1
        return mask

    def _prob_dist(self, s: torch.Tensor, alpha: float) -> torch.Tensor:
        logits = -alpha * torch.log(1 + (s - self.ranks) ** 2)
        gumbel_dist = torch.distributions.Gumbel(0, 1)
        gumbel_vector = gumbel_dist.sample((self.rank_max + 1 - self.rank_min,)).to(self.weight.device)
        return torch.softmax(logits + self.tau * gumbel_vector, dim=0)

    def set_mask(self, s: torch.Tensor, alpha: Optional[float] = None, index: int = 0) -> None:
        self.s = s
        self.alpha = alpha
        self.index = index

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the mask and compute the forward pass.
        """
        if self.s is not None:
            if self.alpha is not None:
                p = self._prob_dist(self.s, self.alpha).to(self.weight.device)
                mask = self._create_mask(p)
                self.log_matrix_to_wandb(mask.cpu().detach().numpy())
            else:
                mask = self._create_discrete_mask()
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight
        return torch.nn.functional.linear(input_tensor, masked_weight, self.bias)
