import abc
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import fast_simplification
import numpy as np
import torch
import trimesh


class TorchMeshContainer:
    r"""Compatability utility structure for NVDiffRast rendering and pytorch-kinematics.

    When given meshes, this structure stores vertex positions, and respectively faces, in a
    single concatenated tensor. Indices (lower and upper) for accessing individual meshes in
    this tensor are stored in a lookup dictionary.
    """

    __slots__ = [
        "_mesh_names",
        "_vertices",  # tensor of shape (B, N, 4) -> homogeneous coordinates
        "_faces",  # tensor of shape (B, N, 3)
        "_per_mesh_vertex_count",
        "_per_mesh_face_count",
        "_lower_vertex_index_lookup",
        "_upper_vertex_index_lookup",
        "_lower_face_index_lookup",
        "_upper_face_index_lookup",
        "_batch_size",
        "_device",
    ]

    def __init__(
        self,
        mesh_paths: Dict[str, str],
        batch_size: int = 1,
        device: torch.device = "cuda",
        target_reduction: float = 0.0,
    ) -> None:
        self._mesh_names = []
        self._vertices = []
        self._faces = []
        self._per_mesh_vertex_count = OrderedDict()
        self._per_mesh_face_count = OrderedDict()
        self._lower_vertex_index_lookup = {}
        self._upper_vertex_index_lookup = {}
        self._lower_face_index_lookup = {}
        self._upper_face_index_lookup = {}

        # load meshes
        self._populate_meshes(mesh_paths, device, target_reduction)

        # add batch dim
        self._batch_size = batch_size
        self._vertices = self._vertices.unsqueeze(0).repeat(self._batch_size, 1, 1)

        # populate index lookups
        self._populate_index_lookups()

        self._device = device

    @abc.abstractmethod
    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        return trimesh.load(mesh_path)

    @abc.abstractmethod
    def _populate_meshes(
        self,
        mesh_paths: Dict[str, str],
        device: torch.device = "cuda",
        target_reduction: float = 0.0,
    ) -> None:
        offset = 0
        for mesh_name, mesh_path in mesh_paths.items():
            # populate mesh names
            self._mesh_names.append(mesh_name)

            # load mesh
            m = self._load_mesh(mesh_path)

            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)

            vertices = m.vertices
            faces = m.faces

            vertices, faces = fast_simplification.simplify(
                points=vertices,
                triangles=faces,
                target_reduction=target_reduction,
            )

            # populate mesh vertex count
            self._per_mesh_vertex_count[mesh_name] = len(vertices)

            # populate vertices
            self._vertices.append(
                torch.tensor(vertices, dtype=torch.float32, device=device)
            )
            self._vertices[-1] = torch.cat(
                [
                    self._vertices[-1],
                    torch.ones_like(self._vertices[-1][:, :1]),
                ],
                dim=1,
            )  # (x,y,z) -> (x,y,z,1): homogeneous coordinates

            # populate mesh face count
            self._per_mesh_face_count[mesh_name] = len(faces)

            # populate faces (also add an offset to the point ids)
            self._faces.append(
                torch.add(
                    torch.tensor(faces, dtype=torch.int32, device=device),
                    offset,
                )
            )
            offset += len(vertices)

        self._vertices = torch.cat(self._vertices, dim=0)
        self._faces = torch.cat(self._faces, dim=0)

    def _populate_index_lookups(self) -> None:
        if len(self._mesh_names) == 0:
            raise ValueError("No meshes loaded.")
        if len(self._per_mesh_vertex_count) == 0:
            raise ValueError("No vertex counts populated.")
        if len(self._per_mesh_face_count) == 0:
            raise ValueError("No face counts populated.")
        # create index lookup
        # crucial: self._per_mesh_vertex_count sorted same as self._vertices! Same for faces.
        running_vertex_index = 0
        running_face_index = 0
        for mesh_name in self._mesh_names:
            # vertex index lookup
            self._lower_vertex_index_lookup[mesh_name] = running_vertex_index
            running_vertex_index += self._per_mesh_vertex_count[mesh_name]
            self._upper_vertex_index_lookup[mesh_name] = running_vertex_index

            # face index lookup
            self._lower_face_index_lookup[mesh_name] = running_face_index
            running_face_index += self._per_mesh_face_count[mesh_name]
            self._upper_face_index_lookup[mesh_name] = running_face_index

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
    def lower_vertex_index_lookup(self) -> Dict[str, int]:
        return self._lower_vertex_index_lookup

    @property
    def upper_vertex_index_lookup(self) -> Dict[str, int]:
        return self._upper_vertex_index_lookup

    @property
    def lower_face_index_lookup(self) -> Dict[str, int]:
        return self._lower_face_index_lookup

    @property
    def upper_face_index_lookup(self) -> Dict[str, int]:
        return self._upper_face_index_lookup

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

    __slots__ = [
        "_intrinsics",
        "_extrinsics",
        "_resolution",
        "_ht_optical",
        "_device",
        "_name",
    ]

    def __init__(
        self,
        resolution: Tuple[int, int],
        intrinsics: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        extrinsics: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        device: torch.device = "cuda",
        name: str = "camera",
    ) -> None:
        if intrinsics is None:
            intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        if extrinsics is None:
            extrinsics = torch.eye(4, dtype=torch.float32, device=device)
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics).float().to(device=device)
        if isinstance(extrinsics, np.ndarray):
            extrinsics = torch.from_numpy(extrinsics).float().to(device=device)
        if (intrinsics.shape != (3, 3)) and (
            intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3)
        ):
            raise ValueError(
                f"Intrinsics must be a 3x3 or Bx3x3 matrix. Got shape {intrinsics.shape} of dim {intrinsics.dim()}."
            )
        if (extrinsics.shape != (4, 4)) and (
            extrinsics.dim() != 3 or extrinsics.shape[-2:] != (4, 4)
        ):
            raise ValueError(
                f"Extrinsics must be a 4x4 or Bx4x4 matrix. Got shape {extrinsics.shape} of dim {extrinsics.dim()}."
            )

        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._resolution = resolution
        self._device = device
        ht_optical_shape = (
            (1,) + extrinsics.shape[-2:]
            if extrinsics.dim() == 3
            else extrinsics.shape[-2:]
        )
        self._ht_optical = torch.zeros(
            ht_optical_shape, dtype=torch.float32, device=device
        )
        self._ht_optical[..., 0, 2] = (
            1.0  # OpenCV-oriented optical frame, in quaternions: [0.5, -0.5, 0.5, -0.5] (w, x, y, z)
        )
        self._ht_optical[..., 1, 0] = -1.0
        self._ht_optical[..., 2, 1] = -1.0
        self._ht_optical[..., 3, 3] = 1.0
        self._name = name

    @abc.abstractmethod
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
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Tuple[int, int]) -> None:
        self._resolution = resolution

    @property
    def height(self) -> int:
        return self._resolution[0]

    @property
    def width(self) -> int:
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

    __slots__ = ["_perspective_projection", "_zmin", "_zmax"]

    def __init__(
        self,
        resolution: Tuple[int, int],
        intrinsics: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        extrinsics: Optional[Union[torch.FloatTensor, np.ndarray]] = None,
        zmin: float = 0.1,
        zmax: float = 100.0,
        device: torch.device = "cuda",
    ) -> None:
        super().__init__(resolution, intrinsics, extrinsics, device)

        # build perspective projection matrix
        self._zmin = zmin
        self._zmax = zmax

        if (
            self._intrinsics.ndim == 2
        ):  # dimension check compatible with numpy and torch
            self._perspective_projection = torch.zeros(4, 4, device=device)
        elif self._intrinsics.ndim == 3:
            self._perspective_projection = torch.zeros(
                self._intrinsics.shape[0], 4, 4, device=device
            )
        else:
            raise ValueError("Intrinsics must be a 3x3 or Bx3x3 matrix.")

        self._perspective_projection[..., 0, 0] = (
            2.0 * self._intrinsics[..., 0, 0] / self.width
        )
        self._perspective_projection[..., 1, 1] = (
            2.0 * self._intrinsics[..., 1, 1] / self.height
        )
        self._perspective_projection[..., 0, 2] = (
            2.0 * self._intrinsics[..., 0, 2] / self.width - 1.0
        )
        self._perspective_projection[..., 1, 2] = (
            2.0 * self._intrinsics[..., 1, 2] / self.height - 1.0
        )
        self._perspective_projection[..., 2, 2] = (zmax + zmin) / (zmax - zmin)
        self._perspective_projection[..., 2, 3] = 2.0 * zmax * zmin / (zmin - zmax)
        self._perspective_projection[..., 3, 2] = 1.0

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
