import os

import trimesh
from trimesh.exchange.export import export_obj


def export_dae_to_obj_mtl(
    input_dae: str,
    output_path: str,
    obj_name: str,
    mtl_name: str,
) -> None:
    r"""Export a DAE file to OBJ and MTL files.

    Args:
        input_dae (str): Path to the DAE file.
        output_path (str): Path to the output directory.
        obj_name (str): Name of the output OBJ file.
        mtl_name (str): Name of the output MTL file.
    """
    mesh = trimesh.load(input_dae)
    obj_content, texture = export_obj(mesh, return_texture=True, mtl_name=mtl_name)

    # write output
    with open(os.path.join(output_path, obj_name), "w") as f:
        f.write(obj_content)

    for texture_name, texture_content in texture.items():
        with open(os.path.join(output_path, texture_name), "wb") as f:
            f.write(texture_content)
