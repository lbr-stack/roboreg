import torch

from roboreg.registration.image.config import CSRegConfig, DRRegConfig
from roboreg.registration.image.objectives import RenderingObjective
from roboreg.registration.image.request import MonocularRequest, StereoRequest
from roboreg.registration.result import RegistrationResult


class MonocularDiffRendRegistration:
    def __init__(
        self,
        config: DRRegConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)

    def __call__(
        self,
        request: MonocularRequest,
    ) -> RegistrationResult:
        pass


class StereoDiffRendRegistration:
    def __init__(
        self,
        config: DRRegConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)

    def __call__(self, request: StereoRequest) -> RegistrationResult:
        pass


class CameraSwarmRegistration:
    def __init__(
        self,
        config: CSRegConfig,
        objective: RenderingObjective,
        device: torch.device | str = "cuda",
    ) -> None:
        self._config = config
        self._objective = objective
        self._device = torch.device(device)

    def __call__(self, request: MonocularRequest) -> RegistrationResult:
        pass
