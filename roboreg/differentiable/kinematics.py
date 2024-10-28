from typing import Dict

import pytorch_kinematics as pk
import torch

from roboreg.io import URDFParser


class TorchKinematics:
    __slots__ = [
        "_root_link_name",
        "_end_link_name",
        "_chain",
        "_mesh_origins_lookup",
        "_device",
    ]

    def __init__(
        self,
        urdf_parser: URDFParser,
        root_link_name: str,
        end_link_name: str,
        device: torch.device = "cuda",
    ) -> None:
        self._root_link_name = root_link_name
        self._end_link_name = end_link_name
        self._chain = self._build_serial_chain_from_urdf(
            urdf_parser.urdf,
            root_link_name=self._root_link_name,
            end_link_name=self._end_link_name,
        )

        self._mesh_origins_lookup = urdf_parser.mesh_origins(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        self._mesh_origins_lookup = {
            key: torch.from_numpy(value).to(device=device, dtype=torch.float32)
            for key, value in self._mesh_origins_lookup.items()
        }

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
        for link_name in self._mesh_origins_lookup:
            self._mesh_origins_lookup[link_name] = self._mesh_origins_lookup[
                link_name
            ].to(device=device)
        self._device = device

    def mesh_forward_kinematics(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Computes forward kinematics and returns corresponding homogeneous transformations.
        Corrects for mesh offsets. Meshes that are tranformed by the returned transformation appear physically correct.
        """
        ht_lookup = {
            key: value.get_matrix() @ self._mesh_origins_lookup[key]
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
