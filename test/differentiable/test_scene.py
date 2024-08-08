import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from roboreg import differentiable as rrd
from roboreg.io import find_files
from roboreg.util import overlay_mask


class TestScene:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        root_link_name: str = "link_0",
        end_link_name: str = "link_7",
        camera_requires_grad: bool = False,
        data_prefix: str = "test/data/lbr_med7",
        recording_prefix: str = "zed2i/stereo_data",
    ) -> None:
        prefix = os.path.join(data_prefix, recording_prefix)

        # set camera names
        self.camera_names = ["left", "right"]

        # instantiate cameras and load masks
        self.masks = {}
        self.images = {}
        for camera_name in self.camera_names:
            self.masks[camera_name] = [
                cv2.imread(os.path.join(prefix, file), cv2.IMREAD_GRAYSCALE)
                for file in find_files(prefix, f"{camera_name}_mask_*.png")
            ]

            self.images[camera_name] = [
                cv2.imread(os.path.join(prefix, file))
                for file in find_files(prefix, f"{camera_name}_img_*.png")
            ]

        # load joint states
        self.joint_states = [
            np.load(os.path.join(prefix, file))
            for file in find_files(prefix, "joint_state_*.npy")
        ]

        # test for equal length
        for camera_name in self.camera_names:
            if len(self.masks[camera_name]) != len(self.images[camera_name]) or len(
                self.masks[camera_name]
            ) != len(self.joint_states):
                raise ValueError(
                    f"Number of masks, images, and joint states do not match for camera '{camera_name}'."
                )

        # to tensors
        for camera_name in self.camera_names:
            self.masks[camera_name] = (
                torch.tensor(
                    np.array(self.masks[camera_name]),
                    dtype=torch.float32,
                    device=device,
                )
                / 255.0
            ).unsqueeze(-1)

        self.joint_states = torch.tensor(
            np.array(self.joint_states), dtype=torch.float32, device=device
        )

        # instantiates camera info and extrinsics files
        camera_info_files = {
            camera_name: os.path.join(prefix, f"{camera_name}_camera_info.yaml")
            for camera_name in self.camera_names
        }
        extrinsics_files = {
            "left": os.path.join(prefix, "HT_hydra_robust.npy"),
            "right": os.path.join(
                prefix,
                "HT_right_to_left.npy",
            ),
        }

        # instantiate scene
        self.scene = rrd.robot_scene_factory(
            device=device,
            batch_size=self.joint_states.shape[0],
            ros_package="lbr_description",
            xacro_path="urdf/med7/med7.xacro",
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            camera_info_files=camera_info_files,
            extrinsics_files=extrinsics_files,
        )

        # enable gradient tracking
        self.scene.cameras["left"].extrinsics.requires_grad = camera_requires_grad


def test_multi_config_stereo_view() -> None:
    test_scene = TestScene(camera_requires_grad=False)

    # configure robot joint states
    test_scene.scene.configure_robot_joint_states(q=test_scene.joint_states)

    # render all camera views
    all_renders = {
        "left": test_scene.scene.observe_from("left"),
        "right": test_scene.scene.observe_from(
            "right", reference_transform=test_scene.scene.cameras["left"].extrinsics
        ),
    }

    # show overlays
    for camera_name, renders in all_renders.items():
        for render, image in zip(renders, test_scene.images[camera_name]):
            render = render.squeeze().cpu().numpy()
            overlay = overlay_mask(
                image,
                (render * 255.0).astype(np.uint8),
                scale=1.0,
            )

            cv2.imshow(camera_name, overlay)
            cv2.waitKey(0)


def test_multi_config_stereo_view_pose_optimization() -> None:
    test_scene = TestScene(camera_requires_grad=True)

    # configure robot joint states
    test_scene.scene.configure_robot_joint_states(q=test_scene.joint_states)

    # instantiante optimizer
    optimizer = torch.optim.SGD([test_scene.scene.cameras["left"].extrinsics], lr=0.001)
    metric = torch.nn.BCELoss()

    initial_right_extrinsics = test_scene.scene.cameras["right"].extrinsics.clone()

    best_loss = float("inf")
    best_extrinsics = test_scene.scene.cameras["left"].extrinsics.clone()
    for _ in tqdm(range(200)):
        # render all camera views
        all_renders = {
            "left": test_scene.scene.observe_from("left"),
            "right": test_scene.scene.observe_from(
                "right", reference_transform=test_scene.scene.cameras["left"].extrinsics
            ),
        }

        # compute loss
        loss = 0.0
        for camera_name in all_renders.keys():
            loss += metric(all_renders[camera_name], test_scene.masks[camera_name])

        if loss < best_loss:
            best_loss = loss
            best_extrinsics = test_scene.scene.cameras["left"].extrinsics.clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show an overlay for a camera
        overlays = []
        for camera_name in test_scene.scene.cameras.keys():
            render = all_renders[camera_name][0].squeeze().detach().cpu().numpy()
            image = test_scene.images[camera_name][0]
            overlays.append(
                overlay_mask(
                    image,
                    (render * 255.0).astype(np.uint8),
                    scale=1.0,
                )
            )
        cv2.imshow("overlays", cv2.resize(np.hstack(overlays), (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

    # expect right intrinsics to be un-changed
    if not torch.allclose(
        initial_right_extrinsics,
        test_scene.scene.cameras["right"].extrinsics,
        atol=1e-4,
    ):
        raise ValueError("Right extrinsics changed during optimization.")

    # reset to best extrinsics and re-render
    test_scene.scene.cameras["left"].extrinsics = best_extrinsics

    with torch.no_grad():
        all_renders = {
            "left": test_scene.scene.observe_from("left"),
            "right": test_scene.scene.observe_from(
                "right", reference_transform=test_scene.scene.cameras["left"].extrinsics
            ),
        }

    # show overlays
    for camera_name, renders in all_renders.items():
        for render, image in zip(renders, test_scene.images[camera_name]):
            render = render.squeeze().detach().cpu().numpy()
            overlay = overlay_mask(
                image,
                (render * 255.0).astype(np.uint8),
                scale=1.0,
            )

            cv2.imshow(camera_name, overlay)
            cv2.waitKey(0)


if __name__ == "__main__":
    # test_multi_config_stereo_view()
    test_multi_config_stereo_view_pose_optimization()
