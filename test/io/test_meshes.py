import numpy as np

from roboreg.io import load_mesh, load_meshes


def test_load_mesh() -> None:
    path = "test/assets/lbr_med7_r800/description/meshes/visual/link_0.dae"
    mesh = load_mesh(path=path)
    assert mesh.vertices.shape[-1] == 3, "Expected vertices of shape Nx3."
    assert mesh.faces.shape[-1] == 3, "Expected faces of shape Nx3."
    assert type(mesh.vertices) == np.ndarray, "Expected vertices to be numpy ndarray."
    assert type(mesh.faces) == np.ndarray, "Expected faces to be numpy ndarray."

    path = "test/assets/lbr_med7_r800/description/meshes/collision/link_0.stl"
    mesh = load_mesh(path=path)
    assert mesh.vertices.shape[-1] == 3, "Expected vertices of shape Nx3."
    assert mesh.faces.shape[-1] == 3, "Expected faces of shape Nx3."
    assert type(mesh.vertices) == np.ndarray, "Expected vertices to be numpy ndarray."
    assert type(mesh.faces) == np.ndarray, "Expected faces to be numpy ndarray."

    path = "link/to/no/file"
    try:
        _ = load_mesh(path)
    except FileNotFoundError:
        pass


def test_load_meshes() -> None:
    paths = {
        "link_0": "test/assets/lbr_med7_r800/description/meshes/collision/link_0.stl",
        "link_1": "test/assets/lbr_med7_r800/description/meshes/collision/link_1.stl",
    }
    meshes = load_meshes(paths=paths)
    assert paths.keys() == meshes.keys(), "Expected same keys."
    assert (
        meshes[list(paths.keys())[0]].vertices.shape[-1] == 3
    ), "Expected vertices of shape Nx3."
    assert (
        meshes[list(paths.keys())[0]].faces.shape[-1] == 3
    ), "Expected faces of shape Nx3."


if __name__ == "__main__":
    import os
    import sys

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_load_mesh()
    test_load_meshes()
