from typing import List

import torch
import trimesh


class TorchRobotMesh:
    _vertices: torch.FloatTensor  # tensor of shape (N, 3)
    _faces: torch.IntTensor  # tensor of shape (N, 3)
    _per_link_vertex_count: List[int]
    _lower_indices: List[int]
    _upper_indices: List[int]

    def __init__(self, mesh_paths: List[str], device: torch.device = "cuda") -> None:
        self._vertices = []
        self._faces = []
        self._per_link_vertex_count = []
        self._lower_indices = []
        self._upper_indices = []
        offset = 0
        for mesh_path in mesh_paths:
            m = trimesh.load(mesh_path)
            self._vertices.append(
                torch.tensor(m.vertices, dtype=torch.float32, device=device)
            )
            # (x,y,z) -> (x,y,z,1)
            self._vertices[-1] = torch.cat(
                [
                    self._vertices[-1],
                    torch.ones_like(self._vertices[-1][:, :1]),
                ],
                dim=1,
            )
            self._per_link_vertex_count.append(len(m.vertices))
            self._faces.append(
                torch.add(
                    torch.tensor(m.faces, dtype=torch.int32, device=device),
                    offset,
                )
            )
            offset += len(m.vertices)
        self._vertices = torch.cat(self._vertices, dim=0)
        self._faces = torch.cat(self._faces, dim=0)

        # add batch dim
        self._vertices = self._vertices.unsqueeze(0)

        for i in range(len(self._per_link_vertex_count)):
            if i == 0:
                self._lower_indices.append(0)
                self._upper_indices.append(self._per_link_vertex_count[i])
            else:
                self._lower_indices.append(self._upper_indices[i - 1])
                self._upper_indices.append(
                    self._upper_indices[i - 1] + self._per_link_vertex_count[i]
                )

    @property
    def vertices(self) -> torch.FloatTensor:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: torch.FloatTensor) -> None:
        self._vertices = vertices

    @property
    def faces(self) -> torch.IntTensor:
        return self._faces

    @property
    def per_link_vertex_count(self) -> List[torch.IntTensor]:
        return self._per_link_vertex_count

    @property
    def lower_indices(self) -> List[int]:
        return self._lower_indices

    @property
    def upper_indices(self) -> List[int]:
        return self._upper_indices

    def set_link_vertices(self, idx: int, vertices: torch.FloatTensor) -> None:
        self._vertices[
            :,
            self._lower_indices[idx] : self._upper_indices[idx],
        ] = vertices

    def get_link_vertices(self, idx: int) -> torch.FloatTensor:
        return self._vertices[
            :,
            self._lower_indices[idx] : self._upper_indices[idx],
        ]
