import torch
import numpy as np
from abc import ABC, abstractmethod


class StepSampler(ABC):
    def __init__(
        self,
        num_diffusion_steps: int
    ) -> None:
        self.num_diffusion_steps = num_diffusion_steps

    @property
    @abstractmethod
    def weights(self) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        w = self.weights
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def state_dict(self) -> dict:
        """Return persistent sampler state for exact training resume."""

        return {
            "sampler_class": type(self).__name__,
            "num_diffusion_steps": int(self.num_diffusion_steps),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore persistent sampler state, validating the sampler contract."""

        if state.get("sampler_class") != type(self).__name__:
            raise ValueError(
                f"Sampler class mismatch: checkpoint has {state.get('sampler_class')!r}, "
                f"current sampler is {type(self).__name__!r}."
            )
        if int(state.get("num_diffusion_steps", -1)) != self.num_diffusion_steps:
            raise ValueError(
                "Sampler num_diffusion_steps does not match the checkpoint state."
            )


class UniformStepSampler(StepSampler):
    """Uniform sampler for the diffusion steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = np.ones([self.num_diffusion_steps], dtype=np.float64)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights


class ImportanceStepSampler(StepSampler):
    """Resampler based on the second moment of the loss.

    From "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672).
    """

    def __init__(
        self,
        min_history_length: int = 10,
        uniform_prob: float = 0.001,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_history_length = min_history_length
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_diffusion_steps, min_history_length], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_diffusion_steps], dtype=np.int64)

    @property
    def weights(self) -> torch.Tensor:
        if not self._warmed_up():
            return np.ones([self.num_diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update(
        self,
        rs: torch.Tensor,
        losses: torch.Tensor
    ) -> None:
        # The sampler history is intentionally kept in NumPy, matching the
        # Improved-DDPM reference implementation. Convert device tensors once
        # before indexing those arrays so the training path also works on CUDA.
        rs_cpu = rs.detach().cpu().tolist()
        losses_cpu = losses.detach().cpu().tolist()
        for r, loss in zip(rs_cpu, losses_cpu):
            if self._loss_counts[r] == self.min_history_length:
                # Shift out the oldest loss term.
                self._loss_history[r, :-1] = self._loss_history[r, 1:]
                self._loss_history[r, -1] = loss
            else:
                self._loss_history[r, self._loss_counts[r]] = loss
                self._loss_counts[r] += 1

    def diagnostics(self) -> dict:
        """Return read-only warm-up diagnostics for training monitoring."""

        counts = self._loss_counts
        return {
            "warmed_up": bool(self._warmed_up()),
            "min_history_count": int(counts.min()),
            "max_history_count": int(counts.max()),
            "mean_history_count": float(counts.mean()),
            "observed_timesteps_fraction": float(np.mean(counts > 0)),
            "full_history_fraction": float(np.mean(counts == self.min_history_length)),
        }

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "min_history_length": int(self.min_history_length),
                "uniform_prob": float(self.uniform_prob),
                "loss_history": self._loss_history.copy(),
                "loss_counts": self._loss_counts.copy(),
            }
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if int(state.get("min_history_length", -1)) != self.min_history_length:
            raise ValueError(
                "Sampler min_history_length does not match the checkpoint state."
            )
        if not np.isclose(
            float(state.get("uniform_prob", np.nan)),
            self.uniform_prob,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Sampler uniform_prob does not match the checkpoint state.")

        loss_history = np.asarray(state.get("loss_history"), dtype=np.float64)
        loss_counts = np.asarray(state.get("loss_counts"), dtype=np.int64)
        if loss_history.shape != self._loss_history.shape:
            raise ValueError(
                f"Sampler loss_history shape mismatch: {loss_history.shape} != "
                f"{self._loss_history.shape}."
            )
        if loss_counts.shape != self._loss_counts.shape:
            raise ValueError(
                f"Sampler loss_counts shape mismatch: {loss_counts.shape} != "
                f"{self._loss_counts.shape}."
            )
        np.copyto(self._loss_history, loss_history)
        np.copyto(self._loss_counts, loss_counts)

    def _warmed_up(self):
        return (self._loss_counts == self.min_history_length).all()
