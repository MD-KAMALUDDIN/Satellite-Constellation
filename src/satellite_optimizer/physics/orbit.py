from dataclasses import dataclass
import math


EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418


@dataclass(frozen=True)
class CircularOrbit:
    """
    Simplified circular orbit model for V1.

    The orbit is defined by:
    - altitude above Earth
    - inclination
    - RAAN
    - initial orbital phase
    """

    altitude_km: float
    inclination_deg: float
    raan_deg: float
    phase_deg: float

    @property
    def radius_km(self) -> float:
        """Orbital radius from Earth's center."""
        return EARTH_RADIUS_KM + self.altitude_km

    @property
    def inclination_rad(self) -> float:
        return math.radians(self.inclination_deg)

    @property
    def raan_rad(self) -> float:
        return math.radians(self.raan_deg)

    @property
    def phase_rad(self) -> float:
        return math.radians(self.phase_deg)

    @property
    def mean_motion_rad_s(self) -> float:
        """
        Mean orbital angular velocity in radians/second.
        """
        return math.sqrt(
            EARTH_MU_KM3_S2 / self.radius_km**3
        )

    @property
    def period_sec(self) -> float:
        """Orbital period in seconds."""
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def period_min(self) -> float:
        """Orbital period in minutes."""
        return self.period_sec / 60.0

    @property
    def velocity_km_s(self) -> float:
        """Circular orbital velocity in km/s."""
        return math.sqrt(
            EARTH_MU_KM3_S2 / self.radius_km
        )

    def phase_at(self, elapsed_seconds: float) -> float:
        """
        Orbital phase after elapsed time.

        Returns radians in the range [0, 2π).
        """
        phase = (
            self.phase_rad
            + self.mean_motion_rad_s * elapsed_seconds
        )

        return phase % (2.0 * math.pi)