from dataclasses import dataclass

from satellite_optimizer.models.mission import MissionConstraints
from satellite_optimizer.physics.geometry import (
    GroundPoint,
    LookGeometry,
    calculate_look_geometry,
    create_ground_point,
)
from satellite_optimizer.physics.orbit import CircularOrbit
from satellite_optimizer.physics.propagation import (
    propagate_circular_orbit,
)


@dataclass(frozen=True)
class VisibilityResult:
    """Result of a satellite-to-target visibility calculation."""

    visible: bool
    geometry: LookGeometry
    elevation_ok: bool
    off_nadir_ok: bool


def check_visibility(
    orbit: CircularOrbit,
    target: GroundPoint,
    elapsed_seconds: float,
    constraints: MissionConstraints,
) -> VisibilityResult:
    """
    Determine whether a satellite can observe a target.

    Hard V1 constraints:
    - minimum elevation
    - maximum off-nadir angle
    """

    eci_position = propagate_circular_orbit(
        orbit,
        elapsed_seconds,
    )

    from satellite_optimizer.physics.geometry import eci_to_ecef

    satellite_ecef = eci_to_ecef(
        x_km=eci_position.x_km,
        y_km=eci_position.y_km,
        z_km=eci_position.z_km,
        elapsed_seconds=elapsed_seconds,
    )

    geometry = calculate_look_geometry(
        satellite_ecef=satellite_ecef,
        target=target,
    )

    elevation_ok = (
        geometry.elevation_deg
        >= constraints.minimum_elevation_deg
    )

    off_nadir_ok = (
        geometry.off_nadir_deg
        <= constraints.maximum_off_nadir_deg
    )

    return VisibilityResult(
        visible=elevation_ok and off_nadir_ok,
        geometry=geometry,
        elevation_ok=elevation_ok,
        off_nadir_ok=off_nadir_ok,
    )


def check_visibility_from_coordinates(
    orbit: CircularOrbit,
    latitude_deg: float,
    longitude_deg: float,
    elapsed_seconds: float,
    constraints: MissionConstraints,
) -> VisibilityResult:
    """
    Convenience wrapper using latitude/longitude directly.
    """

    target = create_ground_point(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
    )

    return check_visibility(
        orbit=orbit,
        target=target,
        elapsed_seconds=elapsed_seconds,
        constraints=constraints,
    )