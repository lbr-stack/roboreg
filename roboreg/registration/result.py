from dataclasses import dataclass
from enum import Enum

import torch


class TerminationReason(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"


@dataclass
class RegistrationResult:
    extrinsics: torch.Tensor
    iterations: int
    termination_reason: TerminationReason
    message: str | None = None

    @property
    def converged(self) -> bool:
        return self.termination_reason == TerminationReason.CONVERGED
