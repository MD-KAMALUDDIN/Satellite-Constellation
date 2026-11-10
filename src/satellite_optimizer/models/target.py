from dataclasses import dataclass


@dataclass
class Target:
    id: str
    name: str

    # Geographic location
    latitude_deg: float
    longitude_deg: float

    # Mission requirements
    priority: float
    observation_duration_sec: float
    data_size_mb: float
    deadline_hours: float