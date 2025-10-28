class AlphaScheduler:
    def step(self, alpha: float) -> float:
        raise NotImplementedError


class LinearAlphaScheduler(AlphaScheduler):
    def __init__(self, alpha_min: float, alpha_max: float, max_steps: int):
        self.increment = (alpha_max - alpha_min) / max_steps if max_steps else 0

    def step(self, alpha: float) -> float:
        return alpha + self.increment


class ExponentialAlphaScheduler(AlphaScheduler):
    def __init__(self, alpha_min: float, alpha_max: float, max_steps: int):
        if alpha_min <= 0:
            raise ValueError("alpha_min must be positive for exponential update type.")
        self.factor = (alpha_max / alpha_min) ** (1 / max_steps) if max_steps else 1.0

    def step(self, alpha: float) -> float:
        return alpha * self.factor


# class CosineAlphaScheduler(AlphaScheduler):
#     def __init__(self, alpha_min, alpha_max, max_steps):
#         self.alpha_min = alpha_min
#         self.alpha_max = alpha_max
#         self.max_steps = max_steps
#         self.current_step = 0

#     def step(self, alpha):
#         import math
#         self.current_step += 1
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * self.current_step / self.max_steps))
#         return self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - cosine_decay)


def get_alpha_scheduler(scheduler_type: str, alpha_min: float, alpha_max: float, max_steps: int) -> AlphaScheduler:
    scheduler_type = scheduler_type.lower()
    schedulers = {
        "linear": LinearAlphaScheduler,
        "exponential": ExponentialAlphaScheduler,
        # "cosine": CosineAlphaScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown alpha_update_type: {scheduler_type}. Available types: {list(schedulers.keys())}")

    return schedulers[scheduler_type](alpha_min, alpha_max, max_steps)
