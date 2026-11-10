import pytest

from satellite_optimizer.physics.orbit import CircularOrbit
from satellite_optimizer.physics.propagation import (
    propagate_circular_orbit,
)


def make_test_orbit() -> CircularOrbit:
    return CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )


def test_position_has_correct_radius():
    orbit = make_test_orbit()

    position = propagate_circular_orbit(
        orbit,
        elapsed_seconds=0.0,
    )

    assert position.magnitude_km == pytest.approx(
        orbit.radius_km,
        abs=1e-6,
    )


def test_position_changes_with_time():
    orbit = make_test_orbit()

    position_0 = propagate_circular_orbit(
        orbit,
        elapsed_seconds=0.0,
    )

    position_60 = propagate_circular_orbit(
        orbit,
        elapsed_seconds=60.0,
    )

    assert position_0.x_km != pytest.approx(
        position_60.x_km
    )


def test_position_returns_after_one_orbit():
    orbit = make_test_orbit()

    position_0 = propagate_circular_orbit(
        orbit,
        elapsed_seconds=0.0,
    )

    position_one_period = propagate_circular_orbit(
        orbit,
        elapsed_seconds=orbit.period_sec,
    )

    assert position_one_period.x_km == pytest.approx(
        position_0.x_km,
        abs=1e-6,
    )

    assert position_one_period.y_km == pytest.approx(
        position_0.y_km,
        abs=1e-6,
    )

    assert position_one_period.z_km == pytest.approx(
        position_0.z_km,
        abs=1e-6,
    )