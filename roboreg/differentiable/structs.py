from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import torch
import trimesh


class TorchMeshContainer:
    r"""Compatability utility structure for NVDiffRast rendering and pytorch-kinematics.

    When given meshes, this structure stores vertex positions, and respectively faces, in a
    single concatenated tensor. Indices (lower and upper) for accessing individual meshes in
    this tensor are stored in a lookup dictionary.
    """

    _mesh_names: List[str]
    _vertices: torch.FloatTensor  # tensor of shape (B, N, 4) -> homogeneous coordinates
    _per_mesh_vertex_count: OrderedDict[str, int]
    _faces: torch.IntTensor  # tensor of shape (B, N, 3)
    _lower_index_lookup: Dict[str, int]
    _upper_index_lookup: Dict[str, int]
    _batch_size: int
    _device: torch.device

    def __init__(
        self,
        mesh_paths: Dict[str, str],
        batch_size: int = 1,
        device: torch.device = "cuda",
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
        self._batch_size = batch_size
        self._vertices = self._vertices.unsqueeze(0).repeat(self._batch_size, 1, 1)

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

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device: torch.device) -> None:
        self._vertices = self._vertices.to(device=device)
        self._faces = self._faces.to(device=device)
        self._device = device


class Camera:
    r"""Simple structure for camera parameters."""

    _intrinsics: torch.FloatTensor
    _extrinsics: torch.FloatTensor
    _resolution: List[int]
    _ht_optical: torch.FloatTensor
    _device: torch.device
    _name: str

    def __init__(
        self,
        intrinsics: Union[torch.FloatTensor, np.ndarray],
        extrinsics: Union[torch.FloatTensor, np.ndarray],
        resolution: List[int],
        device: torch.device = "cuda",
        name: str = "camera",
    ) -> None:
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics).float()
        if isinstance(extrinsics, np.ndarray):
            extrinsics = torch.from_numpy(extrinsics).float()
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._resolution = resolution
        self._ht_optical = torch.tensor(  # OpenCV-oriented optical frame, in quaternions: [0.5, -0.5, 0.5, -0.5] (w, x, y, z)
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.to(device=device)
        self._name = name

    def to(self, device: torch.device) -> None:
        self._intrinsics = self._intrinsics.to(device=device)
        self._extrinsics = self._extrinsics.to(device=device)
        self._ht_optical = self._ht_optical.to(device=device)
        self._device = device

    @property
    def intrinsics(self) -> torch.FloatTensor:
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics: torch.FloatTensor) -> None:
        self._intrinsics = intrinsics

    @property
    def extrinsics(self) -> torch.FloatTensor:
        return self._extrinsics

    @extrinsics.setter
    def extrinsics(self, extrinsics: torch.FloatTensor) -> None:
        self._extrinsics = extrinsics

    @property
    def resolution(self) -> List[int]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: List[int]) -> None:
        self._resolution = resolution

    @property
    def width(self) -> int:
        return self._resolution[0]

    @property
    def height(self) -> int:
        return self._resolution[1]

    @property
    def ht_optical(self) -> torch.FloatTensor:
        return self._ht_optical

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def name(self) -> str:
        return self._name


class VirtualCamera(Camera):
    r"""Simple extension of Camera that adds a perspective projection matrix.

    The perspective projection matrix (with zero skew) is defined as:

    [2*fx/w, 0      , 2*cx/w-1               , 0                            ]
    [0     , 2*fy/h , 2*cy/h-1               , 0                            ]
    [0     , 0      , (zmax+zmin)/(zmax-zmin), 2*(z_max*z_min)/(z_min-z_max)]
    [0     , 0      , 1                      , 0                            ]

    where:
    - `fx`, `fy`: focal lengths in pixels
    - `cx`, `cy`: principal point in pixels
    - `w`, `h`: image width and height in pixels
    - `zmin`, `zmax`: near and far clipping planes

    Further reading:
    - https://sightations.wordpress.com/2010/08/03/simulating-calibrated-cameras-in-opengl/
    - http://www.songho.ca/opengl/gl_projectionmatrix.html
    - http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    - https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ
    """

    _perspective_projection: torch.FloatTensor
    _zmin: float
    _zmax: float

    def __init__(
        self,
        intrinsics: Union[torch.FloatTensor, np.ndarray],
        extrinsics: Union[torch.FloatTensor, np.ndarray],
        resolution: List[int],
        zmin: float = 0.1,
        zmax: float = 100.0,
        device: torch.device = "cuda",
    ) -> None:
        height, width = resolution
        self._zmin = zmin
        self._zmax = zmax
        self._perspective_projection = torch.tensor(
            [
                [
                    2.0 * intrinsics[0, 0] / width,
                    0.0,
                    2.0 * intrinsics[0, 2] / width - 1.0,
                    0.0,
                ],
                [
                    0.0,
                    2.0 * intrinsics[1, 1] / height,
                    2.0 * intrinsics[1, 2] / height - 1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    (zmax + zmin) / (zmax - zmin),
                    2.0 * zmax * zmin / (zmin - zmax),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        super().__init__(intrinsics, extrinsics, resolution, device)

    def to(self, device: torch.device) -> None:
        self._perspective_projection = self._perspective_projection.to(device=device)
        super().to(device=device)

    @property
    def perspective_projection(self) -> torch.FloatTensor:
        return self._perspective_projection

    @property
    def zmin(self) -> float:
        return self._zmin

    @property
    def zmax(self) -> float:
        return self._zmax
