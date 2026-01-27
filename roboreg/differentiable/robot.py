from typing import Union

import torch

from roboreg.io import URDFParser

from .kinematics import TorchKinematics
from .structs import TorchMeshContainer


class Robot:
    __slots__ = ["_mesh_container", "_kinematics", "_configured_vertices", "_device"]

    def __init__(
        self,
        mesh_container: TorchMeshContainer,
        kinematics: TorchKinematics,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        self._mesh_container = mesh_container
        self._kinematics = kinematics
        self._configured_vertices = self.mesh_container.vertices.clone()
        self._device = torch.device(device) if isinstance(device, str) else device
        self.to(device=self._device)

    @classmethod
    def from_urdf_parser(
        cls,
        urdf_parser: URDFParser,
        root_link_name: str,
        end_link_name: str,
        collision: bool = False,
        batch_size: int = 1,
        device: Union[torch.device, str] = "cuda",
        target_reduction: float = 0.0,
    ) -> "Robot":
        from roboreg.io import apply_mesh_origins, load_meshes, simplify_meshes

        # parse data from URDF
        mesh_paths = urdf_parser.ros_package_mesh_paths(
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision,
        )
        mesh_origins = urdf_parser.mesh_origins(
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision,
        )

        # load and preprocess meshes
        meshes = load_meshes(paths=mesh_paths)
        meshes = simplify_meshes(
            meshes=meshes,
            target_reduction=target_reduction,
        )
        meshes = apply_mesh_origins(meshes=meshes, origins=mesh_origins)

        # configure this robot
        mesh_container = TorchMeshContainer(
            meshes=meshes,
            batch_size=batch_size,
            device=device,
        )

        kinematics = TorchKinematics(
            urdf=urdf_parser.urdf,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            device=device,
        )

        return cls(mesh_container=mesh_container, kinematics=kinematics, device=device)

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
