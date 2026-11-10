from satellite_optimizer.models.mission import MissionConstraints
from satellite_optimizer.physics.geometry import create_ground_point
from satellite_optimizer.physics.orbit import CircularOrbit
from satellite_optimizer.physics.visibility import check_visibility


def make_orbit() -> CircularOrbit:
    return CircularOrbit(
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
    )


def make_constraints() -> MissionConstraints:
    return MissionConstraints(
        minimum_elevation_deg=10.0,
        minimum_battery_reserve_percent=20.0,
        maximum_off_nadir_deg=30.0,
    )


def test_visibility_result_contains_geometry():
    result = check_visibility(
        orbit=make_orbit(),
        target=create_ground_point(
            latitude_deg=0.0,
            longitude_deg=0.0,
        ),
        elapsed_seconds=0.0,
        constraints=make_constraints(),
    )

    assert result.geometry.range_km > 0.0
    assert isinstance(result.visible, bool)


def test_elevation_constraint_is_applied():
    constraints = MissionConstraints(
        minimum_elevation_deg=100.0,
        minimum_battery_reserve_percent=20.0,
        maximum_off_nadir_deg=30.0,
    )

    result = check_visibility(
        orbit=make_orbit(),
        target=create_ground_point(
            latitude_deg=0.0,
            longitude_deg=0.0,
        ),
        elapsed_seconds=0.0,
        constraints=constraints,
    )

    assert result.elevation_ok is False
    assert result.visible is False


def test_off_nadir_constraint_is_applied():
    constraints = MissionConstraints(
        minimum_elevation_deg=-90.0,
        minimum_battery_reserve_percent=20.0,
        maximum_off_nadir_deg=0.0,
    )

    result = check_visibility(
        orbit=make_orbit(),
        target=create_ground_point(
            latitude_deg=17.385,
            longitude_deg=78.4867,
        ),
        elapsed_seconds=0.0,
        constraints=constraints,
    )

    assert result.geometry.off_nadir_deg > 0.0
    assert result.off_nadir_ok is False
    assert result.visible is False