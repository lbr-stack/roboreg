from typing import Union

import torch

from .kinematics import TorchKinematics
from .structs import TorchMeshContainer


class Robot:
    __slots__ = ["_mesh_container", "_kinematics", "_configured_vertices", "_device"]

    def __init__(
        self,
        mesh_container: TorchMeshContainer,
        kinematics: TorchKinematics,
    ) -> None:
        self._mesh_container = mesh_container
        self._kinematics = kinematics
        self._configured_vertices = self.mesh_container.vertices.clone()
        if mesh_container.device != kinematics.device:
            raise ValueError(
                "Mesh container and kinematics must be on the same device."
            )
        self._device = mesh_container.device

    def configure(
        self, q: torch.FloatTensor, ht_root: torch.FloatTensor = None
    ) -> None:
        if self._kinematics.chain.n_joints != q.shape[-1]:
            raise ValueError(
                f"Expected joint states of shape {self._kinematics.chain.n_joints}, got {q.shape[-1]}."
            )
        if q.shape[0] != self._mesh_container.batch_size:
            raise ValueError(
                f"Batch size mismatch. Meshes: {self._mesh_container.batch_size}, joint states: {q.shape[0]}."
            )
        if ht_root is None:
            ht_root = torch.eye(4, device=self._device).unsqueeze(0)
        ht_target_lookup = self._kinematics.forward_kinematics(q)
        self._configured_vertices = self.mesh_container.vertices.clone()
        for link_name, ht in ht_target_lookup.items():
            self._configured_vertices[
                :,
                self.mesh_container.lower_vertex_index_lookup[
                    link_name
                ] : self.mesh_container.upper_vertex_index_lookup[link_name],
            ] = torch.matmul(
                torch.matmul(
                    self._configured_vertices[
                        :,
                        self.mesh_container.lower_vertex_index_lookup[
                            link_name
                        ] : self.mesh_container.upper_vertex_index_lookup[link_name],
                    ],
                    ht.transpose(-1, -2),
                ),
                ht_root.transpose(-1, -2),
            )

    def to(self, device: Union[torch.device, str]) -> None:
        self._mesh_container.to(device=device)
        self._kinematics.to(device=device)
        self._configured_vertices = self._configured_vertices.to(device=device)
        self._device = torch.device(device) if isinstance(device, str) else device

    @property
    def kinematics(self) -> TorchKinematics:
        return self._kinematics

    @property
    def mesh_container(self) -> TorchMeshContainer:
        return self._mesh_container

    @property
    def configured_vertices(self) -> torch.FloatTensor:
        return self._configured_vertices

    @property
    def device(self) -> torch.device:
        return self._device
