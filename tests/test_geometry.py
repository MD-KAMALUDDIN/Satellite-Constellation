import pytest

from satellite_optimizer.physics.geometry import (
    create_ground_point,
    eci_to_ecef,
    geodetic_to_ecef,
)


def test_equator_prime_meridian():
    point = geodetic_to_ecef(
        latitude_deg=0.0,
        longitude_deg=0.0,
    )

    assert point.x_km == pytest.approx(6378.137, abs=1e-3)
    assert point.y_km == pytest.approx(0.0, abs=1e-6)
    assert point.z_km == pytest.approx(0.0, abs=1e-6)


def test_north_pole():
    point = geodetic_to_ecef(
        latitude_deg=90.0,
        longitude_deg=0.0,
    )

    assert point.x_km == pytest.approx(0.0, abs=1e-6)
    assert point.y_km == pytest.approx(0.0, abs=1e-6)
    assert point.z_km == pytest.approx(
        6356.752,
        abs=0.01,
    )


def test_ground_point_creation():
    point = create_ground_point(
        latitude_deg=17.385,
        longitude_deg=78.4867,
    )

    assert point.latitude_deg == pytest.approx(17.385)
    assert point.longitude_deg == pytest.approx(78.4867)
    assert point.ecef.magnitude_km > 6300.0


def test_eci_to_ecef_at_zero_time():
    ecef = eci_to_ecef(
        x_km=7000.0,
        y_km=1000.0,
        z_km=500.0,
        elapsed_seconds=0.0,
    )

    assert ecef.x_km == pytest.approx(7000.0)
    assert ecef.y_km == pytest.approx(1000.0)
    assert ecef.z_km == pytest.approx(500.0)


def test_eci_to_ecef_preserves_radius():
    eci = eci_to_ecef(
        x_km=6000.0,
        y_km=3000.0,
        z_km=1000.0,
        elapsed_seconds=5000.0,
    )

    expected_radius = (
        6000.0**2
        + 3000.0**2
        + 1000.0**2
    ) ** 0.5

    assert eci.magnitude_km == pytest.approx(
        expected_radius,
        abs=1e-6,
    )