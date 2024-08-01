from collections import OrderedDict
from typing import Dict, List

import torch
import trimesh


class TorchMeshContainer:
    _mesh_names: List[str]
    _vertices: torch.FloatTensor  # tensor of shape (N, 3)
    _per_mesh_vertex_count: OrderedDict[str, int]
    _faces: torch.IntTensor  # tensor of shape (N, 3)
    _lower_index_lookup: Dict[str, int]
    _upper_index_lookup: Dict[str, int]
    _device: torch.device

    def __init__(
        self, mesh_paths: Dict[str, str], device: torch.device = "cuda"
    ) -> None:
        self._mesh_names = []
        self._vertices = []
        self._per_mesh_vertex_count = OrderedDict()
        self._faces = []
        self._lower_index_lookup = {}
        self._upper_index_lookup = {}

        offset = 0
        for mesh_name, mesh_path in mesh_paths.items():
            # populate mesh names
            self._mesh_names.append(mesh_name)

            # load mesh
            m = trimesh.load(mesh_path)

            # populate mesh vertex count
            self._per_mesh_vertex_count[mesh_name] = len(m.vertices)

            # populate vertices
            self._vertices.append(
                torch.tensor(m.vertices, dtype=torch.float32, device=device)
            )
            self._vertices[-1] = torch.cat(
                [
                    self._vertices[-1],
                    torch.ones_like(self._vertices[-1][:, :1]),
                ],
                dim=1,
            )  # (x,y,z) -> (x,y,z,1): homogeneous coordinates

            # populate faces (also add an offset to the point ids)
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

        # create index lookup
        # crucial: self._per_mesh_vertex_count sorted same as self._vertices!
        index = 0
        for mesh_name, vertex_count in self._per_mesh_vertex_count.items():
            self._lower_index_lookup[mesh_name] = index
            index += vertex_count
            self._upper_index_lookup[mesh_name] = index

        self._device = device

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
    def per_mesh_vertex_count(self) -> OrderedDict[str, torch.IntTensor]:
        return self._per_mesh_vertex_count

    @property
    def lower_index_lookup(self) -> Dict[str, int]:
        return self._lower_index_lookup

    @property
    def upper_index_lookup(self) -> Dict[str, int]:
        return self._upper_index_lookup

    @property
    def mesh_names(self) -> List[str]:
        return self._mesh_names

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> None:
        self._vertices = self._vertices.to(device=device)
        self._faces = self._faces.to(device=device)
        self._device = device

    def set_mesh_vertices(self, mesh_name: str, vertices: torch.FloatTensor) -> None:
        r"""Utility setter for easier access to vertices by mesh."""
        self._vertices[
            :,
            self._lower_index_lookup[mesh_name] : self._upper_index_lookup[mesh_name],
        ] = vertices

    def get_mesh_vertices(self, mesh_name: str) -> torch.FloatTensor:
        r"""Utility getter for easier access to vertices by mesh."""
        return self._vertices[
            :,
            self._lower_index_lookup[mesh_name] : self._upper_index_lookup[mesh_name],
        ]

    def transform_mesh(self, ht: torch.FloatTensor, mesh_name: str) -> None:
        if mesh_name not in self._mesh_names:
            raise ValueError(f"Mesh name {mesh_name} not found in mesh container.")
        self.set_mesh_vertices(
            mesh_name=mesh_name,
            vertices=torch.matmul(
                self.get_mesh_vertices(mesh_name=mesh_name), ht.transpose(-1, -2)
            ),
        )


class Camera:
    _intrinsic: torch.FloatTensor
    _extrinsics: torch.FloatTensor
    _resolution: List[int]
    _device: torch.device

    def __init__(
        self,
        intrinsics: torch.FloatTensor,
        extrinsics: torch.FloatTensor,
        resolution: List[int],
        device: torch.device = "cuda",
    ) -> None:
        self._intrinsic = intrinsics
        self._extrinsics = extrinsics
        self._resolution = resolution
        self.to(device=device)

    def to(self, device: torch.device) -> None:
        self._intrinsic = self._intrinsic.to(device=device)
        self._extrinsics = self._extrinsics.to(device=device)
        self._device = device

    @property
    def intrinsic(self) -> torch.FloatTensor:
        return self._intrinsic

    @property
    def extrinsics(self) -> torch.FloatTensor:
        return self._extrinsics

    @property
    def resolution(self) -> List[int]:
        return self._resolution

    @property
    def device(self) -> torch.device:
        return self._device
