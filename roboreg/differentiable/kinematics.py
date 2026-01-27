from typing import Dict, Union

import pytorch_kinematics as pk
import torch


class TorchKinematics:
    __slots__ = [
        "_root_link_name",
        "_end_link_name",
        "_chain",
        "_device",
    ]

    def __init__(
        self,
        urdf: str,
        root_link_name: str,
        end_link_name: str,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        self._root_link_name = root_link_name
        self._end_link_name = end_link_name
        self._chain = self._build_serial_chain_from_urdf(
            urdf=urdf,
            root_link_name=self._root_link_name,
            end_link_name=self._end_link_name,
        )
        self._device = torch.device(device) if isinstance(device, str) else device
        self.to(device=self._device)

    def _build_serial_chain_from_urdf(
        self, urdf: str, root_link_name: str, end_link_name: str
    ) -> pk.SerialChain:
        return pk.build_serial_chain_from_urdf(
            urdf, end_link_name=end_link_name, root_link_name=root_link_name
        )

    def to(self, device: Union[torch.device, str]) -> None:
        self._chain.to(device=device)
        self._device = torch.device(device) if isinstance(device, str) else device

    def forward_kinematics(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        ht_lookup = {
            key: value.get_matrix()
            for key, value in self._chain.forward_kinematics(q, end_only=False).items()
        }
        return ht_lookup

    @property
    def root_link_name(self) -> str:
        return self._root_link_name

    @property
    def end_link_name(self) -> str:
        return self._end_link_name

    @property
    def chain(self) -> pk.SerialChain:
        return self._chain

    @property
    def device(self) -> torch.device:
        return self._device
