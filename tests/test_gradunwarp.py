"""Tests for vendor-neutral gradient nonlinearity correction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mrdistortion import GradientCoefficients, Gradunwarp
from mrdistortion._gradunwarp import _evaluate_harmonics


class StubAccessor:
    """A coefficient source that is not a file, as an integration would be.

    The scanner's own accessor reads an MRD header and lives with the server
    that speaks MRD. What belongs here is only that ``from_file`` takes one at
    all, which is what the protocol promises.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read_coefficients(self) -> str:
        return self._payload


def _zero_coefficients(order: int = 1) -> GradientCoefficients:
    values = np.zeros((3, order + 1, order + 1))
    return GradientCoefficients("unnormalized", values, values)


def _dat_payload(*, scale10: float = 0.0) -> str:
    lines = ["GRADWARPTYPE 1"]
    for axis, multiplier in zip("XYZ", (1.0, 2.0, 3.0), strict=True):
        for order in range(1, 11):
            value = multiplier * order if order < 10 else scale10
            lines.append(f"SCALE{axis}{order} {value}")
    lines.append("DELTA 0.0")
    return "\n".join(lines)


def test_dat_conversion_extends_beta_converter_to_ninth_order() -> None:
    coefficients = GradientCoefficients.from_file(_dat_payload())

    assert coefficients.max_order == 9
    assert coefficients.alpha[0, 3, 1] == pytest.approx(3.0 * 2.0 / 3.0)
    assert coefficients.beta[1, 7, 1] == pytest.approx((2.0 * 7.0) * 16.0 / 7.0)
    assert coefficients.alpha[2, 9, 0] == pytest.approx((3.0 * 9.0) * -128.0)
    assert coefficients.basis == "unnormalized"
    assert "alpha" not in repr(coefficients)
    assert "beta" not in repr(coefficients)


def test_dat_conversion_rejects_undocumented_nonzero_tenth_order() -> None:
    with pytest.raises(ValueError, match=r"SCALE.*10"):
        GradientCoefficients.from_file(_dat_payload(scale10=1e-12))


def test_coefficient_path_and_serialized_text_are_equivalent(tmp_path) -> None:
    path = tmp_path / "coefficients.dat"
    path.write_text(_dat_payload())

    from_path = GradientCoefficients.from_file(path)
    from_text = GradientCoefficients.from_file(_dat_payload())

    np.testing.assert_array_equal(from_path.alpha, from_text.alpha)
    np.testing.assert_array_equal(from_path.beta, from_text.beta)


def test_normalized_coefficient_parser_reads_radius_and_general_degrees() -> None:
    payload = """
    0.25 m = R0, lnorm = 4
      1 A( 3, 0) -0.01 z
    101 A( 3, 1)  0.02 x
    201 B( 5, 3) -0.03 y
    """

    coefficients = GradientCoefficients.from_file(payload)

    assert coefficients.reference_radius_mm == 250.0
    assert coefficients.basis == "normalized"
    assert coefficients.alpha[2, 3, 0] == -0.01
    assert coefficients.alpha[0, 3, 1] == 0.02
    assert coefficients.beta[1, 5, 3] == -0.03


def _mrd_header(table: str | None = "", encodings: int = 1):
    """An MRD XML header, with as much of it as this correction reads."""
    user = SimpleNamespace(
        userParameterString=[SimpleNamespace(name="GradientCoefficients", value=table)]
        if table is not None
        else []
    )
    encoding = [
        SimpleNamespace(
            reconSpace=SimpleNamespace(
                matrixSize=SimpleNamespace(x=16 + index, y=12, z=4),
                fieldOfView_mm=SimpleNamespace(x=240.0, y=180.0, z=60.0),
            )
        )
        for index in range(encodings)
    ]
    return SimpleNamespace(encoding=encoding, userParameters=user)


def _mrd_acquisition(encoding: int = 0):
    return SimpleNamespace(
        encoding_space_ref=encoding,
        read_dir=(-1.0, 0.0, 0.0),
        phase_dir=(0.0, 1.0, 0.0),
        slice_dir=(0.0, 0.0, 1.0),
        position=(10.0, -20.0, 4.0),
    )


def test_from_mrd_reads_the_recon_space_of_the_encoding_the_acquisition_names():
    """The matrix and field of view come from the encoding, not the acquisition."""
    correct = Gradunwarp.from_mrd(
        _mrd_header(_dat_payload(), encodings=3), _mrd_acquisition(encoding=2)
    )

    # MRD states x along the readout; the array is indexed (slice, line, column).
    assert correct.shape == (4, 12, 18)
    np.testing.assert_allclose(correct.target_grid.shape, (4, 12, 18, 3))


def test_from_mrd_takes_the_orientation_and_centre_from_the_acquisition() -> None:
    correct = Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition())

    # Columns follow (slice, line, column), so read_dir is the last of them.
    np.testing.assert_allclose(
        correct._acquired.direction,
        np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
    )
    np.testing.assert_allclose(correct._acquired.center_mm, (10.0, -20.0, 4.0))


def test_from_mrd_reads_the_coil_table_out_of_the_user_parameters() -> None:
    correct = Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition())

    assert correct.coefficients.max_order == 9


def test_from_mrd_accepts_a_table_a_stream_does_not_carry() -> None:
    """A site whose coefficients travel separately passes them in."""
    correct = Gradunwarp.from_mrd(
        _mrd_header(table=None),
        _mrd_acquisition(),
        coefficients=StubAccessor(_dat_payload()),
    )

    assert correct.coefficients.max_order == 9


def test_from_mrd_refuses_a_stream_with_no_table_anywhere() -> None:
    with pytest.raises(ValueError, match="No gradient coefficients"):
        Gradunwarp.from_mrd(_mrd_header(table=None), _mrd_acquisition())


def test_from_mrd_refuses_an_encoding_the_header_does_not_have() -> None:
    with pytest.raises(ValueError, match="names encoding 2"):
        Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition(encoding=2))


def test_affine_geometry_preserves_oblique_physical_grid() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((1.2, 1.5, 2.0))
    affine[:3, 3] = (4.0, -5.0, 6.0)
    shape = (8, 10, 12)

    correct = Gradunwarp.from_affine(_zero_coefficients(), affine, shape)
    index = (2, 3, 4)

    expected = (affine @ np.append(np.asarray(index, dtype=float), 1.0))[:3]
    np.testing.assert_allclose(correct.target_grid[index], expected, atol=1e-10)
    np.testing.assert_allclose(correct._acquired.scanner_to_indices(expected), index)
    np.testing.assert_allclose(correct._acquired.fov_mm, (9.6, 15.0, 24.0))


def test_zero_coefficients_are_identity_for_real_and_complex_batches() -> None:
    angle = np.deg2rad(20.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
            (0.0, 0.0),
        )
    )
    correct = Gradunwarp(_zero_coefficients(), (7, 8), (70.0, 80.0), direction)
    rng = np.random.default_rng(4)
    image = rng.normal(size=(2, *correct.shape)).astype(np.float32)
    complex_image = image + 1j * image[::-1]

    np.testing.assert_allclose(correct(image), image, atol=2e-5)
    np.testing.assert_allclose(correct(complex_image), complex_image, atol=2e-5)
    np.testing.assert_allclose(correct.jacobian_grid, 1.0)


def test_a_target_grid_corrects_and_reslices_in_one_step() -> None:
    direction = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    correct = Gradunwarp(
        _zero_coefficients(),
        (9, 9),
        (90.0, 90.0),
        direction,
        target_shape=(17, 17),
        target_fov_mm=(80.0, 80.0),
        jacobian=False,
    )
    acquired = Gradunwarp(_zero_coefficients(), (9, 9), (90.0, 90.0), direction)
    physical = acquired.target_grid
    image = (2.0 * physical[..., 0] - 3.0 * physical[..., 1] + 7.0).astype(np.float32)

    result = correct(image)
    expected = acquired._acquired.scanner_to_indices(correct.target_grid)

    assert result.shape == (17, 17)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(correct.source_grid, expected, atol=1e-12)


def test_full_3d_jacobian_is_used_for_a_2d_plane() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    correct = Gradunwarp(
        coefficients,
        (11, 12),
        (110.0, 120.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )

    # P_1^1(cos(theta)) cos(phi) * r = -x. The output-to-source
    # convention therefore yields source_x = 1.01*x and determinant 1.01.
    np.testing.assert_allclose(correct.jacobian_grid, 1.01, atol=2e-8)


def test_cartesian_dat_recurrence_matches_second_order_solid_harmonics() -> None:
    alpha = np.zeros((3, 3, 3))
    beta = np.zeros_like(alpha)
    alpha[0, 2, 1] = 0.2
    beta[1, 2, 1] = -0.3
    alpha[2, 2, 0] = 0.4
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    coordinates = np.asarray(((20.0, -30.0, 40.0), (-10.0, 50.0, -20.0)))
    x, y, z = (coordinates / 10.0).T
    radius_squared = x * x + y * y + z * z
    expected = 10.0 * np.stack(
        (
            0.2 * (-3.0 * x * z),
            -0.3 * (-3.0 * y * z),
            0.4 * (3.0 * z * z - radius_squared) / 2.0,
        ),
        axis=-1,
    )

    np.testing.assert_allclose(
        _evaluate_harmonics(coefficients, coordinates),
        expected,
        atol=1e-12,
    )


def test_compiled_3d_jacobian_uses_all_three_physical_derivatives() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    beta[1, 1, 1] = 0.02
    alpha[2, 1, 0] = 0.03
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    angle = np.deg2rad(23.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    determinant = Gradunwarp(
        coefficients, (9, 10, 11), (90.0, 100.0, 110.0), direction
    ).jacobian_grid

    np.testing.assert_allclose(
        determinant[1:-1, 1:-1, 1:-1],
        1.01 * 1.02 * 0.97,
        atol=1e-12,
    )


def test_the_constructor_accepts_a_coefficient_accessor_directly() -> None:
    correct = Gradunwarp(
        StubAccessor(_dat_payload()),
        (8, 9),
        (80.0, 90.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )

    assert correct.coefficients.max_order == 9


#: What a GE console states for its body coil, in the syntax it states it in.
#: A reconstruction runs off the scanner, so the description travels rather
#: than a path to a file on it.
EMITTED_COIL_DESCRIPTION = (
    "GRADWARPTYPE 1\n"
    "SCALEX1 0.000000000e+00\n"
    "SCALEX2 0.000000000e+00\n"
    "SCALEX3 -1.674469968e-04\n"
    "SCALEX4 0.000000000e+00\n"
    "SCALEX5 -8.157819309e-08\n"
    "SCALEX6 0.000000000e+00\n"
    "SCALEX7 0.000000000e+00\n"
    "SCALEX8 0.000000000e+00\n"
    "SCALEX9 0.000000000e+00\n"
    "SCALEX10 0.000000000e+00\n"
    "SCALEY1 0.000000000e+00\n"
    "SCALEY2 0.000000000e+00\n"
    "SCALEY3 -1.426626986e-04\n"
    "SCALEY4 0.000000000e+00\n"
    "SCALEY5 -8.702937038e-08\n"
    "SCALEY6 0.000000000e+00\n"
    "SCALEY7 0.000000000e+00\n"
    "SCALEY8 0.000000000e+00\n"
    "SCALEY9 0.000000000e+00\n"
    "SCALEY10 0.000000000e+00\n"
    "SCALEZ1 0.000000000e+00\n"
    "SCALEZ2 0.000000000e+00\n"
    "SCALEZ3 -1.136897990e-04\n"
    "SCALEZ4 0.000000000e+00\n"
    "SCALEZ5 -1.055270982e-08\n"
    "SCALEZ6 0.000000000e+00\n"
    "SCALEZ7 0.000000000e+00\n"
    "SCALEZ8 0.000000000e+00\n"
    "SCALEZ9 0.000000000e+00\n"
    "SCALEZ10 0.000000000e+00\n"
    "DELTA 0.000000000e+00\n"
)


def test_a_real_coil_description_parses_with_the_documented_factors():
    coefficients = GradientCoefficients.from_file(EMITTED_COIL_DESCRIPTION)

    assert coefficients.basis == "unnormalized"
    assert coefficients.max_order == 9
    # Third and fifth order on each axis, the only terms this coil states.
    assert coefficients.alpha[0, 3, 1] == pytest.approx(-1.674470e-4 * 2.0 / 3.0)
    assert coefficients.beta[1, 3, 1] == pytest.approx(-1.426627e-4 * 2.0 / 3.0)
    assert coefficients.alpha[2, 3, 0] == pytest.approx(-1.136898e-4 * -2.0)
    assert coefficients.alpha[0, 5, 1] == pytest.approx(-8.157819e-8 * 8.0 / 15.0)
    assert coefficients.beta[1, 5, 1] == pytest.approx(-8.702937e-8 * 8.0 / 15.0)
    assert coefficients.alpha[2, 5, 0] == pytest.approx(-1.055271e-8 * -8.0)
