from typing import List, Optional

import cv2
import numpy as np
import pyvista
import torch


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    mode: str = "r",
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
    scale: float = 2.0,
) -> np.ndarray:
    r"""Overlay mask on image.

    Args:
        img: Image of shape HxWx3.
        mask: Mask of shape HxW.
        mode: Color mode. "r", "g", or "b".
        alpha: Alpha value for the mask.
        scale: Scale factor for the image.

    Returns:
        Mask overlayed on image.
    """
    colored_mask = None
    if mode == "r":
        colored_mask = np.stack(
            [np.zeros_like(mask), np.zeros_like(mask), mask], axis=2
        )
    elif mode == "g":
        colored_mask = np.stack(
            [np.zeros_like(mask), mask, np.zeros_like(mask)], axis=2
        )
    elif mode == "b":
        colored_mask = np.stack(
            [mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2
        )
    else:
        raise ValueError("Mode must be r, g, or b.")

    overlay_img_mask = cv2.addWeighted(img, alpha, colored_mask, beta, gamma)
    # resize by scale
    overlay_img_mask = cv2.resize(
        overlay_img_mask,
        [int(size * scale) for size in overlay_img_mask.shape[:2][::-1]],
    )
    return overlay_img_mask


class RegistrationVisualizer(object):
    def colorize_mesh(self, mesh_vertices: List[np.ndarray]) -> List[np.ndarray]:
        mesh_colors = []

        for i in range(len(mesh_vertices)):
            mesh_color = np.array(
                [
                    [
                        0.5 + (len(mesh_vertices) - i - 1) / len(mesh_vertices) / 2.0,
                        0.5,
                        0.8,
                        1.0,
                    ]
                ]
            )
            mesh_colors.append(np.tile(mesh_color, (mesh_vertices[i].shape[0], 1)))
        return mesh_colors

    def colorize_observed(
        self, observed_vertices: List[np.ndarray]
    ) -> List[np.ndarray]:
        observed_colors = []

        for i in range(len(observed_vertices)):
            observed_color = np.array(
                [
                    [
                        0.5,
                        0.8,
                        0.5
                        + (len(observed_vertices) - i - 1)
                        / len(observed_vertices)
                        / 2.0,
                        1.0,
                    ]
                ]
            )
            observed_colors.append(
                np.tile(observed_color, (observed_vertices[i].shape[0], 1))
            )
        return observed_colors

    def __call__(
        self,
        mesh_vertices: List[torch.Tensor],
        observed_vertices: List[torch.Tensor],
        HT: Optional[torch.Tensor] = None,
        background_color: List[int] = [0, 0, 0],
    ) -> None:
        plotter = pyvista.Plotter()
        plotter.add_axes()
        plotter.background_color = background_color

        if HT is not None:
            mesh_vertices = [
                torch.mm(HT[:3, :3], mesh_vertex.T).T + HT[:3, 3]
                for mesh_vertex in mesh_vertices
            ]

        # to numpy
        np_mesh_vertices = []
        np_observed_vertices = []
        for i in range(len(mesh_vertices)):
            np_mesh_vertices.append(mesh_vertices[i].cpu().numpy())
            np_observed_vertices.append(observed_vertices[i].cpu().numpy())

        # get colors
        mesh_colors = self.colorize_mesh(np_mesh_vertices)
        observed_colors = self.colorize_observed(np_observed_vertices)

        for mesh_color, mesh_vertex in zip(mesh_colors, np_mesh_vertices):
            plotter.add_points(
                mesh_vertex, scalars=mesh_color, rgba=True, show_scalar_bar=False
            )

        for observed_color, observed_vertex in zip(
            observed_colors, np_observed_vertices
        ):
            plotter.add_points(
                observed_vertex,
                scalars=observed_color,
                rgba=True,
                show_scalar_bar=False,
            )

        plotter.show()
