from dataclasses import dataclass
import math

import numpy as np

from satellite_optimizer.physics.orbit import CircularOrbit


@dataclass(frozen=True)
class ECIPosition:
    """
    Earth-Centered Inertial position in kilometers.
    """

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


def propagate_circular_orbit(
    orbit: CircularOrbit,
    elapsed_seconds: float,
) -> ECIPosition:
    """
    Propagate a simplified circular orbit.

    Returns the satellite's ECI position after the
    specified elapsed time.
    """

    radius = orbit.radius_km
    theta = orbit.phase_at(elapsed_seconds)

    inclination = orbit.inclination_rad
    raan = orbit.raan_rad

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    cos_i = math.cos(inclination)
    sin_i = math.sin(inclination)

    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)

    # Position in orbital plane.
    x_orb = radius * cos_theta
    y_orb = radius * sin_theta

    # Rotate orbital-plane coordinates into ECI.
    x_eci = (
        cos_raan * x_orb
        - sin_raan * y_orb * cos_i
    )

    y_eci = (
        sin_raan * x_orb
        + cos_raan * y_orb * cos_i
    )

    z_eci = y_orb * sin_i

    return ECIPosition(
        x_km=x_eci,
        y_km=y_eci,
        z_km=z_eci,
    )