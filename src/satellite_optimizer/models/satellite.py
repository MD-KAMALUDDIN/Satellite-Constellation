from dataclasses import dataclass


@dataclass
class Satellite:
    id: str
    name: str

    # Orbital parameters
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    phase_deg: float

    # Power
    battery_capacity_wh: float
    initial_battery_wh: float

    # Storage
    storage_capacity_gb: float
    initial_storage_gb: float

    # Attitude control
    slew_rate_deg_per_sec: float