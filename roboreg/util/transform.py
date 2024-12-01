from typing import Optional

import torch


def depth_to_xyz(
    depth: torch.FloatTensor,
    intrinsics: torch.FloatTensor,
    z_min: float = 0.01,
    z_max: float = 2,
    conversion_factor: float = 1.0e3,
) -> torch.FloatTensor:
    r"""Converts a depth image to a point cloud. Note that this function uses the OpenCV convention.

    Args:
        depth (torch.FloatTensor): Depth image of shape HxW / BxHxW / Bx1xHxW.
        intrinsics (torch.FloatTensor): Camera intrinsics of shape 3x3 or Bx3x3.
        z_min (float): Minimum depth value.
        z_max (float): Maximum depth value.
        conversion_factor (float): Conversion factor for depth. Computes z = depth / conversion_factor.

    Returns:
        Point cloud of shape HxWx3 or BxHxWx3.
    """
    if intrinsics.dim() > depth.dim():
        raise ValueError("Expected intrinsics to have fewer dimensions than depth.")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("Expected intrinsics to have shape 3x3 or Bx3x3.")
    if z_min >= z_max:
        raise ValueError("Expected z_min to be less than z_max.")
    if conversion_factor <= 0:
        raise ValueError("Expected conversion_factor to be greater than zero.")
    if z_min <= 0:
        raise ValueError("Expected z_min to be greater than zero.")
    height, width = depth.shape[-2:]
    x = torch.linspace(0, width - 1, width, device=depth.device)
    y = torch.linspace(0, height - 1, height, device=depth.device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    if depth.dim() == 3:
        x = x.unsqueeze(0).expand(depth.shape[0], -1, -1)
        y = y.unsqueeze(0).expand(depth.shape[0], -1, -1)
    z = depth / conversion_factor
    # fill with nans where z_min <= z <= z_max
    z = torch.where((z < z_min) | (z > z_max), torch.nan, z)
    if intrinsics.dim() == 2 and depth.dim() == 3:
        intrinsics = intrinsics.unsqueeze(0).expand(depth.shape[0], -1, -1)
    x = (
        (x - intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1))
        * z
        / intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (y - intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1))
        * z
        / intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    return torch.stack((x, y, z), dim=-1)


def generate_ht_optical(
    batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = "cuda",
) -> torch.Tensor:
    ht_optical = torch.zeros(4, 4, dtype=dtype, device=device)
    if batch_size is not None:
        ht_optical = ht_optical.unsqueeze(0).expand(batch_size, -1, -1)
    ht_optical[..., 0, 2] = (
        1.0  # OpenCV-oriented optical frame, in quaternions: [0.5, -0.5, 0.5, -0.5] (w, x, y, z)
    )
    ht_optical[..., 1, 0] = -1.0
    ht_optical[..., 2, 1] = -1.0
    ht_optical[..., 3, 3] = 1.0
    return ht_optical


def look_at_from_angle(
    eye: torch.Tensor,
    center: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    r"""Given eye, center, and angle, returns the corresponding look-at matrix.

    Args:
        eye (torch.Tensor): The eye position of shape Bx3.
        center (torch.Tensor): The center position of shape Bx3.
        angle (torch.Tensor): The angle in radians of shape Bx1.

    Returns:
        torch.Tensor: The look-at matrix of shape Bx4x4.
    """
    if not eye.device == center.device or not eye.device == angle.device:
        raise ValueError("Expected all tensors to be on the same device.")
    if not eye.dim() == center.dim() == angle.dim() == 2:
        raise ValueError("Expected all tensors to be 2D.")
    if not eye.shape[0] == center.shape[0] == angle.shape[0]:
        raise ValueError("Expected all tensors to have the same batch size.")

    device = eye.device
    batch_size = eye.shape[0]

    up = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(batch_size, 1)

    random_x = (center - eye) / torch.norm(
        center - eye, dim=1, keepdim=True
    )  # camera frame (user has to add optical transform after...)

    random_y = torch.cross(up, random_x, dim=1)
    random_z = torch.cross(random_x, random_y, dim=1)

    # normalize all x,y,z
    random_x = random_x / torch.norm(random_x, dim=1, keepdim=True)
    random_y = random_y / torch.norm(random_y, dim=1, keepdim=True)
    random_z = random_z / torch.norm(random_z, dim=1, keepdim=True)

    random_ht = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    random_ht[:, :3, 0] = random_x
    random_ht[:, :3, 1] = random_y
    random_ht[:, :3, 2] = random_z
    random_ht[:, :3, 3] = eye

    # add a random rotation about the z-axis in [-angle_interval/2, angle_interval/2]
    random_rot = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    angle = angle.squeeze()
    random_rot[:, 0, 0] = torch.cos(angle)
    random_rot[:, 0, 1] = -torch.sin(angle)
    random_rot[:, 1, 0] = torch.sin(angle)
    random_rot[:, 1, 1] = torch.cos(angle)

    return random_ht @ random_rot
