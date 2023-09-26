from typing import List

import torch


class InstanceICP(object):
    HT: torch.Tensor

    def __init__(self) -> None:
        HT = torch.eye(4)

    def __call__(
        self, targets: List[torch.Tensor], sources: List[torch.Tensor]
    ) -> torch.Tensor:
        self._com_init(targets, sources)

        return self.HT

    def _com_init(
        self, targets: List[torch.Tensor], sources: List[torch.Tensor]
    ) -> None:
        targets_com, sources_com = [], []
        for target, source in zip(targets, sources):
            targets_com.append(target.mean(dim=0))
            sources_com.append(source.mean(dim=0))

        targets_com = torch.stack(targets_com)
        sources_com = torch.stack(sources_com)

        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
        X, residuals, rank, singular_values = torch.linalg.lstsq(
            targets_com, sources_com
        )

        print(X)
