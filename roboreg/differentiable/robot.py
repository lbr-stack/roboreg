import torch

from roboreg.io import URDFParser

from .kinematics import TorchKinematics
from .structs import TorchMeshContainer


class Robot(TorchMeshContainer):
    __slots__ = ["_kinematics", "_configured_vertices"]

    def __init__(
        self,
        urdf_parser: URDFParser,
        root_link_name: str,
        end_link_name: str,
        visual: bool = False,
        batch_size: int = 1,
        device: torch.device = "cuda",
        target_reduction: float = 0.0,
    ) -> None:
        super().__init__(
            mesh_paths=urdf_parser.ros_package_mesh_paths(
                root_link_name=root_link_name,
                end_link_name=end_link_name,
                visual=visual,
            ),
            batch_size=batch_size,
            device=device,
            target_reduction=target_reduction,
        )
        self._kinematics = TorchKinematics(
            urdf_parser=urdf_parser,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            device=device,
        )
        self._configured_vertices = self.vertices.clone()

    def configure(
        self, q: torch.FloatTensor, ht_root: torch.FloatTensor = None
    ) -> None:
        if self._kinematics.chain.n_joints != q.shape[-1]:
            raise ValueError(
                f"Expected joint states of shape {self._kinematics.chain.n_joints}, got {q.shape[-1]}."
            )
        if q.shape[0] != self._batch_size:
            raise ValueError(
                f"Batch size mismatch. Meshes: {self._batch_size}, joint states: {q.shape[0]}."
            )
        if ht_root is None:
            ht_root = torch.eye(4, device=self._device).unsqueeze(0)
        ht_target_lookup = self._kinematics.mesh_forward_kinematics(q)
        self._configured_vertices = self.vertices.clone()
        for link_name, ht in ht_target_lookup.items():
            self._configured_vertices[
                :,
                self.lower_vertex_index_lookup[
                    link_name
                ] : self.upper_vertex_index_lookup[link_name],
            ] = torch.matmul(
                torch.matmul(
                    self._configured_vertices[
                        :,
                        self.lower_vertex_index_lookup[
                            link_name
                        ] : self.upper_vertex_index_lookup[link_name],
                    ],
                    ht.transpose(-1, -2),
                ),
                ht_root.transpose(-1, -2),
            )

    @property
    def kinematics(self) -> TorchKinematics:
        return self._kinematics

    @property
    def configured_vertices(self) -> torch.FloatTensor:
        return self._configured_vertices
