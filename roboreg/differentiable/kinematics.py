from typing import Any, Dict

import pytorch_kinematics as pk
import torch


class TorchKinematics:
    _chain: pk.SerialChain

    def __init__(
        self, urdf: str, end_link_name: str, root_link_name: str, device: torch.device
    ) -> None:
        self._end_link_name = end_link_name
        self._root_link_name = root_link_name
        self._chain = self._build_serial_chain_from_urdf(
            urdf, self._end_link_name, self._root_link_name
        )
        self.to(device=device)

    def _build_serial_chain_from_urdf(
        self, urdf: str, end_link_name: str, root_link_name: str
    ) -> pk.SerialChain:
        return pk.build_serial_chain_from_urdf(
            urdf, end_link_name=end_link_name, root_link_name=root_link_name
        )

    def to(self, device: torch.device) -> None:
        self._chain.to(device=device)

    def forward_kinematics(self, q: torch.Tensor) -> Dict[Any, torch.Tensor]:
        transforms_dict = self._chain.forward_kinematics(q, end_only=False)
        return {key: value.get_matrix() for key, value in transforms_dict.items()}

    @property
    def chain(self) -> pk.SerialChain:
        return self._chain

    @property
    def end_link_name(self) -> str:
        return self._end_link_name

    @property
    def root_link_name(self) -> str:
        return self._root_link_name
