from dataclasses import dataclass
from typing import Protocol

import torch

from roboreg.losses import soft_dice_loss
from roboreg.util.mask import mask_distance_transform, mask_exponential_decay


class RenderingObjective(Protocol):
    def validate_targets(self, targets: torch.Tensor) -> None: ...

    def preprocess_targets(self, targets: torch.Tensor) -> torch.Tensor: ...

    def __call__(
        self, preprocessed_targets: torch.Tensor, renders: torch.Tensor
    ) -> torch.Tensor: ...


def _ensure_binary_masks(
    targets: torch.Tensor,
    threshold: float | None,
) -> torch.Tensor:
    if threshold is not None:
        return targets >= threshold

    is_binary = torch.logical_or(
        targets == 0,
        targets == 1,
    )

    if not torch.all(is_binary):
        raise ValueError(
            "Expected binary targets. Set a threshold to convert "
            "probability maps into binary masks."
        )

    return targets.bool()


@dataclass(frozen=True)
class DistanceMapConfig:
    threshold: float | None = None


class DistanceMapObjective:
    r"""Computes the mean squared error between the distance transform of the target mask and the rendered mask.
    Supports binary masks and probability maps as targets. A threshold is required for probability maps.
    """

    def __init__(
        self,
        config: DistanceMapConfig | None = None,
    ) -> None:
        self._config = config or DistanceMapConfig()

    def __post_init__(self) -> None:
        if (
            self._config.threshold is not None
            and self._config.threshold <= 0
            or self._config.threshold >= 1
        ):
            raise ValueError("threshold must be in [0, 1].")

    def validate_targets(self, targets: torch.Tensor) -> None:
        if not torch.all((targets >= 0) & (targets <= 1)):
            raise ValueError("Expected targets in range [0, 1].")
        _ensure_binary_masks(targets, threshold=self._config.threshold)

    def preprocess_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = _ensure_binary_masks(targets, threshold=self._config.threshold)
        targets_np = targets.cpu().numpy()
        distance_maps = [mask_distance_transform(mask) for mask in targets_np]
        return torch.tensor(
            distance_maps, dtype=torch.float32, device=targets.device
        ).unsqueeze(-1)

    def __call__(
        self, preprocessed_targets: torch.Tensor, renders: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((preprocessed_targets - renders) ** 2)


@dataclass(frozen=True)
class ExponentialDecayMaskConfig:
    sigma: float = 2.0
    epsilon: float = 1e-6
    threshold: float | None = None


class ExponentialDecayMaskObjective:
    r"""Computes a soft Dice loss between an exponentially decaying target mask and the rendered mask.
    Supports binary masks and probability maps as targets. A threshold is required for probability maps.
    """

    def __init__(
        self,
        config: ExponentialDecayMaskConfig | None = None,
    ) -> None:
        self._config = config or ExponentialDecayMaskConfig()

    def __post_init__(self) -> None:
        if self._config.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self._config.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if (
            self._config.threshold is not None
            and self._config.threshold <= 0
            or self._config.threshold >= 1
        ):
            raise ValueError("threshold must be in [0, 1].")

    def validate_targets(self, targets: torch.Tensor) -> None:
        if not torch.all((targets >= 0) & (targets <= 1)):
            raise ValueError("Expected targets in range [0, 1].")
        _ensure_binary_masks(targets, threshold=self._config.threshold)

    def preprocess_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = _ensure_binary_masks(targets, threshold=self._config.threshold)
        targets_np = targets.cpu().numpy()
        decay_maps = [
            mask_exponential_decay(mask, sigma=self._config.sigma)
            for mask in targets_np
        ]
        return torch.tensor(
            decay_maps, dtype=torch.float32, device=targets.device
        ).unsqueeze(-1)

    def __call__(
        self, preprocessed_targets: torch.Tensor, renders: torch.Tensor
    ) -> torch.Tensor:
        return soft_dice_loss(
            preprocessed_targets, renders, epsilon=self._config.epsilon
        ).mean()


@dataclass(frozen=True)
class ProbabilityMapConfig:
    epsilon: float = 1e-6


class ProbabilityMapObjective:
    r"""Computes a soft Dice loss between the target probability map and the rendered mask."""

    def __init__(
        self,
        config: ProbabilityMapConfig | None = None,
    ) -> None:
        self._config = config or ProbabilityMapConfig()

    def __post_init__(self) -> None:
        if self._config.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

    def validate_targets(self, targets: torch.Tensor) -> None:
        if not targets.is_floating_point():
            raise ValueError("Expected floating point probability targets.")
        if not torch.all((targets >= 0) & (targets <= 1)):
            raise ValueError("Expected targets in range [0, 1].")

    def preprocess_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets

    def __call__(
        self, preprocessed_targets: torch.Tensor, renders: torch.Tensor
    ) -> torch.Tensor:
        return soft_dice_loss(
            preprocessed_targets, renders, epsilon=self._config.epsilon
        ).mean()
