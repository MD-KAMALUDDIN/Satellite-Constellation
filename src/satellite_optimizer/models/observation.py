from dataclasses import dataclass
from datetime import datetime


@dataclass
class ObservationOpportunity:
    satellite_id: str
    target_id: str

    start_time: datetime
    end_time: datetime

    elevation_deg: float
    off_nadir_deg: float
    slew_angle_deg: float
    slew_time_sec: float

    image_quality: float

    energy_cost_wh: float
    data_size_mb: float

    feasible: bool = True