from dataclasses import dataclass


@dataclass
class Sensor:
    id: str
    name: str

    # Imaging characteristics
    max_off_nadir_deg: float
    observation_power_w: float
    data_rate_mb_per_sec: float