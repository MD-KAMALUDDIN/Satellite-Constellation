import math

import pytest

from satellite_optimizer.physics.orbit import CircularOrbit


def test_circular_orbit_radius():
    orbit = CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )

    assert orbit.radius_km == pytest.approx(6928.137)


def test_circular_orbit_period():
    orbit = CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )

    # 550 km circular orbit should be roughly 95.5 minutes.
    assert orbit.period_min == pytest.approx(95.5, abs=0.5)


def test_circular_orbit_velocity():
    orbit = CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )

    assert orbit.velocity_km_s == pytest.approx(7.59, abs=0.02)


def test_phase_advances_with_time():
    orbit = CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )

    phase = orbit.phase_at(60.0)

    assert 0.0 < phase < 2.0 * math.pi