from dataclasses import dataclass
import math

import numpy as np


EARTH_ROTATION_RATE_RAD_S = 7.2921150e-5


@dataclass(frozen=True)
class ECEFPosition:
    """Earth-Centered, Earth-Fixed position in kilometers."""

    x_km: float
    y_km: float
    z_km: float

    @property
    def magnitude_km(self) -> float:
        return math.sqrt(
            self.x_km**2
            + self.y_km**2
            + self.z_km**2
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x_km, self.y_km, self.z_km],
            dtype=float,
        )


@dataclass(frozen=True)
class GroundPoint:
    """Ground location represented in latitude/longitude and ECEF."""

    latitude_deg: float
    longitude_deg: float
    ecef: ECEFPosition


@dataclass(frozen=True)
class LookGeometry:
    """Satellite-to-ground observation geometry."""

    range_km: float
    elevation_deg: float
    off_nadir_deg: float


def eci_to_ecef(
    x_km: float,
    y_km: float,
    z_km: float,
    elapsed_seconds: float,
) -> ECEFPosition:
    """
    Convert ECI coordinates to ECEF coordinates.

    This V1 model uses a simplified constant Earth rotation rate.
    """

    theta = EARTH_ROTATION_RATE_RAD_S * elapsed_seconds

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_ecef = (
        cos_theta * x_km
        + sin_theta * y_km
    )

    y_ecef = (
        -sin_theta * x_km
        + cos_theta * y_km
    )

    return ECEFPosition(
        x_km=x_ecef,
        y_km=y_ecef,
        z_km=z_km,
    )


def geodetic_to_ecef(
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float = 0.0,
) -> ECEFPosition:
    """
    Convert geodetic latitude/longitude to ECEF.

    Uses the WGS84 ellipsoid.
    """

    semi_major_km = 6378.137
    flattening = 1.0 / 298.257223563
    eccentricity_sq = (
        flattening * (2.0 - flattening)
    )

    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)

    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)

    prime_vertical_radius = (
        semi_major_km
        / math.sqrt(
            1.0
            - eccentricity_sq * sin_lat**2
        )
    )

    x = (
        prime_vertical_radius + altitude_km
    ) * cos_lat * math.cos(longitude)

    y = (
        prime_vertical_radius + altitude_km
    ) * cos_lat * math.sin(longitude)

    z = (
        prime_vertical_radius
        * (1.0 - eccentricity_sq)
        + altitude_km
    ) * sin_lat

    return ECEFPosition(
        x_km=x,
        y_km=y,
        z_km=z,
    )


def create_ground_point(
    latitude_deg: float,
    longitude_deg: float,
) -> GroundPoint:
    """Create an Earth surface point."""

    return GroundPoint(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        ecef=geodetic_to_ecef(
            latitude_deg,
            longitude_deg,
        ),
    )


def calculate_look_geometry(
    satellite_ecef: ECEFPosition,
    target: GroundPoint,
) -> LookGeometry:
    """
    Calculate range, elevation, and off-nadir angle.

    Elevation is measured relative to the local horizon.
    Off-nadir is measured relative to the satellite's
    direction toward Earth's center.
    """

    satellite_vector = satellite_ecef.as_array()
    target_vector = target.ecef.as_array()

    line_of_sight = target_vector - satellite_vector

    range_km = float(np.linalg.norm(line_of_sight))

    if range_km <= 0.0:
        raise ValueError(
            "Satellite and target positions cannot be identical."
        )

    # Local vertical at the target points away from Earth center.
    target_up = target_vector / np.linalg.norm(target_vector)

    los_unit = line_of_sight / range_km

    # Elevation angle above local horizon.
    elevation_rad = math.asin(
    float(np.dot(-los_unit, target_up))
)

    # Satellite nadir direction points toward Earth center.
    nadir = -satellite_vector
    nadir_unit = nadir / np.linalg.norm(nadir)

    # Angle between nadir and line of sight.
    off_nadir_rad = math.acos(
        max(
            -1.0,
            min(
                1.0,
                float(np.dot(nadir_unit, los_unit)),
            ),
        )
    )

    return LookGeometry(
        range_km=range_km,
        elevation_deg=math.degrees(elevation_rad),
        off_nadir_deg=math.degrees(off_nadir_rad),
    )