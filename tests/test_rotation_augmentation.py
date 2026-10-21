"""SO(3), panorama convention, and isolated randomness regression tests."""
import unittest

import numpy as np
import torch


from models.dataset import rotation_augmentation as augmentation


def numpy_rays(height, width):
    latitude = (np.arange(height) + 0.5) * np.pi / height - np.pi / 2
    longitude = (np.arange(width) + 0.5) * 2 * np.pi / width - np.pi
    phi, theta = np.meshgrid(latitude, longitude, indexing='ij')
    return np.stack([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta),
                     -np.sin(phi)], axis=-1)


def numpy_rotate_depth(scalar, matrix):
    """Independent NumPy reference following the archived rotation experiment."""
    height, width = scalar.shape
    old = numpy_rays(height, width) @ matrix
    longitude = np.arctan2(old[..., 1], old[..., 0])
    latitude = np.arctan2(-old[..., 2], np.hypot(old[..., 0], old[..., 1]))
    x = (longitude + np.pi) * width / (2 * np.pi) - 0.5
    y = np.clip((latitude + np.pi / 2) * height / np.pi - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % width, np.minimum(y0 + 1, height - 1)
    x0 %= width
    return ((1 - fy) * ((1 - fx) * scalar[y0, x0] + fx * scalar[y0, x1])
            + fy * ((1 - fx) * scalar[y1, x0] + fx * scalar[y1, x1]))


def geometry(batch=1, height=8, width=16):
    scalar = np.linspace(0.5, 8.5, height * width).reshape(height, width)
    xyz = torch.tensor(scalar[..., None] * numpy_rays(height, width), dtype=torch.float32)
    depth = xyz.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
    locations = torch.tensor([[[1., 2., 3.], [-4., 5., -6.], [0., 0., 0.]]]).repeat(batch, 1, 1)
    return depth, locations


class RotationAugmentationTests(unittest.TestCase):
    def test_configuration_rejects_coercion_and_nonfinite_probability(self):
        for enabled in (0, 1, 'true', None):
            with self.subTest(enabled=enabled), self.assertRaises((TypeError, ValueError)):
                augmentation.RotationAugmentation(enabled, 0.5)
        for probability in (True, '0.5', None, -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(probability=probability), self.assertRaises((TypeError, ValueError)):
                augmentation.RotationAugmentation(True, probability)

    def test_disabled_and_zero_preserve_objects_and_rng(self):
        depth, locations = geometry()
        for enabled, probability in ((False, 1.), (True, 0.)):
            generator = torch.Generator().manual_seed(41)
            state = generator.get_state().clone()
            actual = augmentation.RotationAugmentation(enabled, probability)(depth, locations, generator)
            self.assertIs(actual[0], depth)
            self.assertIs(actual[1], locations)
            self.assertTrue(torch.equal(state, generator.get_state()))

    def test_identity_preserves_geometry(self):
        depth, locations = geometry(2)
        rotated, points = augmentation.rotate_geometry(depth, locations, torch.eye(3).repeat(2, 1, 1))
        torch.testing.assert_close(rotated, depth, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(points, locations, atol=0, rtol=0)

    def test_exact_lattice_rotations_and_location_direction(self):
        depth, locations = geometry()
        scalar = torch.linalg.vector_norm(depth, dim=1)
        rays = torch.tensor(numpy_rays(8, 16), dtype=torch.float32).permute(2, 0, 1)
        cases = [
            ([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], scalar.roll(4, -1), [-2., 1., 3.]),
            ([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], scalar.flip((-2, -1)), [1., -2., -3.]),
            ([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], scalar.flip((-2, -1)).roll(8, -1), [-1., 2., -3.]),
        ]
        for matrix, expected, first_point in cases:
            with self.subTest(matrix=matrix):
                actual, points = augmentation.rotate_geometry(depth, locations, torch.tensor([matrix]))
                torch.testing.assert_close(actual, expected[:, None] * rays, atol=2e-5, rtol=2e-6)
                torch.testing.assert_close(points[0, 0], torch.tensor(first_point))

    def test_longitude_seam_blends_last_and_first_column(self):
        depth, locations = geometry()
        angle = np.pi / 16
        matrix = torch.tensor([[[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]], dtype=torch.float32)
        actual, _ = augmentation.rotate_geometry(depth, locations, matrix)
        scalar = torch.linalg.vector_norm(depth, dim=1)
        expected = (scalar + scalar.roll(1, -1)) / 2
        torch.testing.assert_close(torch.linalg.vector_norm(actual, dim=1), expected, atol=2e-5, rtol=2e-6)

    def test_poles_clamp_to_sampled_latitude(self):
        rays = torch.tensor(numpy_rays(5, 10), dtype=torch.float32).permute(2, 0, 1)
        scalar = torch.arange(1, 6, dtype=torch.float32)[:, None].expand(5, 10)
        depth = (rays * scalar).unsqueeze(0)
        matrix = torch.tensor([[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]])
        actual, _ = augmentation.rotate_geometry(depth, torch.zeros(1, 1, 3), matrix)
        lengths = torch.linalg.vector_norm(actual, dim=1)[0]
        torch.testing.assert_close(lengths[2, 2], torch.tensor(1.), atol=2e-6, rtol=0)
        torch.testing.assert_close(lengths[2, 7], torch.tensor(5.), atol=2e-6, rtol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_general_rotations_match_independent_numpy_reference(self):
        depth, locations = geometry(3, 13, 26)
        matrices = augmentation.sample_rotations(3, depth.device, torch.Generator().manual_seed(77))
        actual, points = augmentation.rotate_geometry(depth, locations, matrices)
        for index in range(3):
            scalar = np.linalg.norm(depth[index].numpy(), axis=0)
            expected_scalar = numpy_rotate_depth(scalar, matrices[index].double().numpy())
            expected = expected_scalar[..., None] * numpy_rays(13, 26)
            np.testing.assert_allclose(actual[index].permute(1, 2, 0).numpy(), expected, atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(points, locations @ matrices.transpose(-1, -2), atol=2e-6, rtol=2e-6)

    def test_locations_preserve_origin_and_pairwise_distances(self):
        depth, locations = geometry(7)
        actual, points = augmentation.RotationAugmentation(True, 1.)(depth, locations, torch.Generator().manual_seed(7))
        torch.testing.assert_close(torch.cdist(points, points), torch.cdist(locations, locations), atol=3e-6, rtol=2e-6)
        torch.testing.assert_close(points.norm(dim=-1), locations.norm(dim=-1), atol=3e-6, rtol=2e-6)
        self.assertTrue(torch.isfinite(actual).all())

    def test_probability_mask_and_unselected_bitwise_unchanged(self):
        depth, locations = geometry(2000, 2, 4)
        original_depth, original_locations = depth.clone(), locations.clone()
        generator = torch.Generator().manual_seed(17)
        expected_mask = torch.rand(2000, generator=torch.Generator().manual_seed(17)) < 0.35
        actual, points = augmentation.RotationAugmentation(True, 0.35)(depth, locations, generator)
        changed = (points != locations).flatten(1).any(dim=1)
        self.assertTrue(torch.equal(changed, expected_mask))
        self.assertLess(abs(changed.float().mean().item() - 0.35), 0.04)
        self.assertTrue(torch.equal(actual[~changed], depth[~changed]))
        self.assertTrue(torch.equal(points[~changed], locations[~changed]))
        self.assertTrue(torch.equal(depth, original_depth))
        self.assertTrue(torch.equal(locations, original_locations))

    def test_local_generator_reproducible_without_global_rng_changes(self):
        depth, locations = geometry(5)
        global_state = torch.get_rng_state().clone()
        transform = augmentation.RotationAugmentation(True, 0.8)
        a = transform(depth, locations, torch.Generator().manual_seed(23))
        b = transform(depth, locations, torch.Generator().manual_seed(23))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_state))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_rotation_samples_are_proper_and_uniform_second_moments(self):
        matrices = augmentation.sample_rotations(30000, torch.device('cpu'), torch.Generator().manual_seed(11))
        torch.testing.assert_close(matrices @ matrices.transpose(-1, -2), torch.eye(3).expand_as(matrices), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(30000), atol=1e-6, rtol=1e-6)
        self.assertLess(matrices.mean(0).abs().max().item(), 0.015)
        torch.testing.assert_close(matrices.square().mean(0), torch.full((3, 3), 1 / 3), atol=0.012, rtol=0)
        direction = matrices[:, :, 0]
        torch.testing.assert_close(direction.T @ direction / len(direction), torch.eye(3) / 3, atol=0.012, rtol=0)

    def test_selected_geometry_uses_float32_no_grad_under_autocast(self):
        depth, locations = geometry(2)
        depth.requires_grad_()
        locations.requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, points = augmentation.RotationAugmentation(True, 1.)(depth, locations, torch.Generator().manual_seed(8))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(points.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(points.requires_grad)


if __name__ == '__main__':
    unittest.main()
