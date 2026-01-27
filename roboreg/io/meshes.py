from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import fast_simplification
import numpy as np
import trimesh


@dataclass
class Mesh:
    r"""Dataclass to hold mesh data."""

    vertices: np.ndarray
    faces: np.ndarray


def load_mesh(path: Union[Path, str]) -> Mesh:
    r"""Reads a mesh file and returns vertices and faces.

    Args:
        path (Union[Path, str]): Path to the mesh file.

    Returns:
        Mesh:
            - Vertices of shape Nx3.
            - Faces of shape Nx3.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file {path} does not exist.")
    m = trimesh.load(path)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    vertices, faces = m.vertices, m.faces
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh is empty: {path.name}")
    return Mesh(vertices=vertices.view(np.ndarray), faces=faces.view(np.ndarray))


def load_meshes(
    paths: Dict[str, Union[Path, str]],
) -> Dict[str, Mesh]:
    r"""Load multiple meshes.

    Args:
        paths (Dict[str, Union[Path, str]]): Mesh names and corresponding paths.

    Returns:
        Dict[str, Mesh]:
            - Mesh name.
            - Mesh vertices of shape Nx3 and faces of shape Nx3.
    """
    return {name: load_mesh(path) for name, path in paths.items()}


def simplify_mesh(mesh: Mesh, target_reduction: float = 0.0) -> Mesh:
    r"""Simplify a mesh.

    Args:
        mesh (Mesh): The mesh to be simplified.
        target_reduction (float): Target reduction in [0, 1]. Zero for no reduction.

    Returns:
        Mesh: The simplified mesh.
    """
    if target_reduction == 0.0:
        return mesh
    if 0.0 > target_reduction > 1.0:
        raise ValueError(
            f"Expected target reduction in [0, 1], got {target_reduction}."
        )
    vertices, faces = fast_simplification.simplify(
        points=mesh.vertices,
        triangles=mesh.faces,
        target_reduction=target_reduction,
    )
    return Mesh(vertices=vertices, faces=faces)


def simplify_meshes(
    meshes: Dict[str, Mesh], target_reduction: float = 0.0
) -> Dict[str, Mesh]:
    f"""Simplify multiple meshes.

    Args:
        meshes (Dict[str, Mesh]): The meshes to be simplified.
        target_reduction (float): The target reduction in [0, 1]. Zero for no reduction.

    Returns:
        Dict[str, Mesh]: The simplified meshes. 
    """
    if target_reduction == 0.0:
        return meshes
    return {
        name: simplify_mesh(mesh, target_reduction=target_reduction)
        for name, mesh in meshes.items()
    }


def apply_mesh_origin(vertices: np.ndarray, origin: np.ndarray) -> np.ndarray:
    r"""Apply a homogeneous transformation to mesh vertices.

    Args:
        vertices (np.ndarray): The mesh vertices of shape Nx3.
        origin (np.ndarray): The mesh origin as a homogeneous transform of shape 4x4.
    Returns:
        np.ndarray: The transformed mesh vertices of shape Nx3.
    """
    return vertices @ origin[:3, :3].T + origin[:3, 3].T


def apply_mesh_origins(
    meshes: Dict[str, Mesh], origins: Dict[str, np.ndarray]
) -> Dict[str, Mesh]:
    r"""Apply mesh origins to multiple meshes.

    Args:
        meshes (Dict[str, Mesh]): The meshes to apply origins to.
        origins (Dict[str, np.ndarray]): The mesh origins.

    Returns:
        Dict[str, Mesh]: The meshes with applied origins.
    """
    for name in meshes.keys():
        if name in origins:
            meshes[name].vertices = apply_mesh_origin(
                meshes[name].vertices, origins[name]
            )
    return meshes
