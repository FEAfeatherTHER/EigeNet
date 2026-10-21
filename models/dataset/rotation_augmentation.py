"""Receiver-centred, uniformly sampled SO(3) augmentation for XYZ panoramas.

Only geometry changes: the caller retains every audio tensor and sample index.
Panorama axes and pixel centres match ``utils.convert_equirect_to_camera_coord``.
"""
from functools import lru_cache
import math
from numbers import Real

import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def _pixel_rays(height, width, device):
    """Return cached H x W x 3 unit rays, with north at positive z."""
    phi = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (math.pi / height) - math.pi / 2
    theta = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2 * math.pi / width) - math.pi
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')
    return torch.stack((phi.cos() * theta.cos(), phi.cos() * theta.sin(), -phi.sin()), dim=-1)


@torch.no_grad()
def sample_rotations(count, device, generator):
    """Sample Haar-uniform proper rotations using a caller-owned generator."""
    device = torch.device(device)
    if not isinstance(generator, torch.Generator):
        raise TypeError('generator must be a torch.Generator')
    with torch.autocast(device_type=device.type, enabled=False):
        quaternion = torch.randn((count, 4), device=device, dtype=torch.float32, generator=generator)
        quaternion = F.normalize(quaternion, dim=-1)
        w, x, y, z = quaternion.unbind(-1)
        return torch.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), dim=-1).reshape(count, 3, 3)


@torch.no_grad()
def rotate_geometry(depth, locations, matrices):
    """Rotate B x 3 x H x W XYZ panoramas and B x N x 3 locations.

    Matrices are B x 3 x 3 active rotations. Inverse panorama sampling uses
    ``output_ray @ R``; locations use ``location @ R.T``. Scalar depth is
    bilinearly sampled with periodic longitude and clamped latitude, then
    projected onto the fixed output pixel rays. All arithmetic is float32.
    """
    if depth.ndim != 4 or depth.shape[1] != 3:
        raise ValueError('depth must have shape [B, 3, H, W]')
    batch, _, height, width = depth.shape
    if locations.ndim != 3 or locations.shape[0] != batch or locations.shape[-1] != 3:
        raise ValueError('locations must have shape [B, N, 3]')
    if matrices.shape != (batch, 3, 3):
        raise ValueError('matrices must have shape [B, 3, 3]')
    if depth.device != locations.device or depth.device != matrices.device:
        raise ValueError('depth, locations, and matrices must share a device')
    if height < 1 or width < 1:
        raise ValueError('panorama dimensions must be positive')
    with torch.autocast(device_type=depth.device.type, enabled=False):
        rays = _pixel_rays(height, width, depth.device)
        matrices = matrices.float()
        # Explicit three-term products stay float32 even when the training
        # entrypoint enables TF32 matmul. Tiny angular errors otherwise move
        # panorama samples across discontinuous room-depth boundaries.
        old_rays = sum(rays[None, ..., axis, None] * matrices[:, None, None, axis, :]
                       for axis in range(3))
        theta = torch.atan2(old_rays[..., 1], old_rays[..., 0])
        phi = torch.atan2(-old_rays[..., 2], torch.linalg.vector_norm(old_rays[..., :2], dim=-1))
        x = (theta + math.pi) * (width / (2 * math.pi)) - 0.5
        y = ((phi + math.pi / 2) * (height / math.pi) - 0.5).clamp(0, height - 1)
        scalar = torch.linalg.vector_norm(depth.float(), dim=1, keepdim=True)
        # Padding one column at each side lets bilinear interpolation cross
        # the longitude seam, including coordinates half a pixel outside.
        periodic = torch.cat((scalar[..., -1:], scalar, scalar[..., :1]), dim=-1)
        grid = torch.stack((2 * (x + 1.5) / (width + 2) - 1,
                            2 * (y + 0.5) / height - 1), dim=-1)
        rotated_scalar = F.grid_sample(periodic, grid, mode='bilinear', padding_mode='border', align_corners=False)
        rotated_depth = rotated_scalar * rays.permute(2, 0, 1).unsqueeze(0)
        points = locations.float()
        rotated_locations = sum(points[..., axis, None] * matrices[:, None, :, axis]
                                for axis in range(3))
        return rotated_depth, rotated_locations


class RotationAugmentation:
    """Independently rotate each sample with the configured probability."""

    def __init__(self, enabled: bool, probability: float):
        if type(enabled) is not bool:
            raise TypeError('enabled must be a bool')
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise TypeError('probability must be a real number')
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('probability must be finite and in [0, 1]')
        self.enabled = enabled
        self.probability = float(probability)

    @torch.no_grad()
    def __call__(self, depth, locations, generator):
        if not self.enabled or self.probability == 0:
            return depth, locations
        if not isinstance(generator, torch.Generator):
            raise TypeError('generator must be a torch.Generator')
        with torch.autocast(device_type=depth.device.type, enabled=False):
            selected = torch.rand(depth.shape[0], device=depth.device, dtype=torch.float32,
                                  generator=generator) < self.probability
            indices = selected.nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                return depth, locations
            matrices = sample_rotations(indices.numel(), depth.device, generator)
            rotated_depth, rotated_locations = rotate_geometry(depth[indices], locations[indices], matrices)
            output_depth, output_locations = depth.float().clone(), locations.float().clone()
            output_depth[indices] = rotated_depth
            output_locations[indices] = rotated_locations
            return output_depth, output_locations
