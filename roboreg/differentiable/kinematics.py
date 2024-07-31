from typing import Dict

import pytorch_kinematics as pk
import torch


class TorchKinematics:
    _root_link_name: str
    _end_link_name: str
    _chain: pk.SerialChain
    _global_joint_offset: Dict[str, torch.Tensor]

    def __init__(
        self, urdf: str, root_link_name: str, end_link_name: str, device: torch.device
    ) -> None:
        self._root_link_name = root_link_name
        self._end_link_name = end_link_name
        self._chain = self._build_serial_chain_from_urdf(
            urdf, root_link_name=self._root_link_name, end_link_name=self._end_link_name
        )

        # populate global joint offset
        self._global_joint_offset = {}
        ht_joint_global = torch.eye(
            4, device=self._chain.device, dtype=self._chain.dtype
        )
        for link_name in self._chain.get_link_names():
            ht_joint_global = (
                ht_joint_global
                @ self._chain.joint_offsets[self._chain.get_frame_indices(link_name)]
            )
            self._global_joint_offset[link_name] = ht_joint_global

        # default move to device
        self.to(device=device)

    def _build_serial_chain_from_urdf(
        self, urdf: str, root_link_name: str, end_link_name: str
    ) -> pk.SerialChain:
        return pk.build_serial_chain_from_urdf(
            urdf, end_link_name=end_link_name, root_link_name=root_link_name
        )

    def to(self, device: torch.device) -> None:
        self._chain.to(device=device)
        for link_name in self._global_joint_offset:
            self._global_joint_offset[link_name] = self._global_joint_offset[
                link_name
            ].to(device=device)

    def mesh_forward_kinematics(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Computes forward kinematics and returns corresponding homogeneous transformations.
        Corrects for mesh offsets. Meshes that are tranformed by the returned transformation appear physically correct.
        """
        ht_lookup = {
            key: value.get_matrix() @ self._global_joint_offset[key].inverse()
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
