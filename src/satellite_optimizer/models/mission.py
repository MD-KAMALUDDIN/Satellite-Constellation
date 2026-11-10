from dataclasses import dataclass


@dataclass
class MissionConstraints:
    minimum_elevation_deg: float
    minimum_battery_reserve_percent: float
    maximum_off_nadir_deg: float


@dataclass
class MissionScoringWeights:
    task_value_weight: float
    image_quality_weight: float
    future_value_weight: float
    energy_weight: float
    slew_weight: float
    data_weight: float


@dataclass
class QuantumConfig:
    algorithm: str
    default_depth: int
    shots: int
    max_candidates: int
    max_iterations: int = 50


@dataclass
class SimulationConfig:
    duration_hours: float
    time_step_seconds: float


@dataclass
class ExecutionConfig:
    default_backend: str
    allow_ibm_simulator: bool
    allow_ibm_hardware: bool


@dataclass
class Mission:
    name: str
    simulation: SimulationConfig
    constraints: MissionConstraints
    scoring: MissionScoringWeights
    quantum: QuantumConfig
    execution: ExecutionConfig