import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from roboreg.diff_rend.util import export_dae_to_obj_mtl


def test_dae_to_obj_exporter() -> None:
    input_dae = "test/data/lbr_med7/mesh/link_0.dae"

    output_path = "test/data/lbr_med7/mesh"

    obj_name = "link_0.obj"
    mtl_name = "link_0.mtl"

    export_dae_to_obj_mtl(
        input_dae,
        output_path,
        obj_name,
        mtl_name,
    )


if __name__ == "__main__":
    test_dae_to_obj_exporter()
