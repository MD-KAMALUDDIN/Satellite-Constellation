from pathlib import Path
from typing import Any

import yaml

from satellite_optimizer.models.mission import (
    ExecutionConfig,
    Mission,
    MissionConstraints,
    MissionScoringWeights,
    QuantumConfig,
    SimulationConfig,
)
from satellite_optimizer.models.satellite import Satellite
from satellite_optimizer.models.target import Target


# Absolute path to the project's config/ directory, regardless of cwd.
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class ConfigLoader:
    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.config_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure in {path}")

        return data

    def load_satellites(self) -> list[Satellite]:
        data = self._load_yaml("satellites.yaml")

        return [
            Satellite(**satellite)
            for satellite in data["satellites"]
        ]

    def load_targets(self) -> list[Target]:
        data = self._load_yaml("targets.yaml")

        return [
            Target(**target)
            for target in data["targets"]
        ]

    def load_mission(self) -> Mission:
        data = self._load_yaml("mission.yaml")["mission"]

        return Mission(
            name=data["name"],
            simulation=SimulationConfig(**data["simulation"]),
            constraints=MissionConstraints(**data["constraints"]),
            scoring=MissionScoringWeights(**data["scoring"]),
            quantum=QuantumConfig(**data["quantum"]),
            execution=ExecutionConfig(**data["execution"]),
        )

    def load_all(self) -> tuple[list[Satellite], list[Target], Mission]:
        satellites = self.load_satellites()
        targets = self.load_targets()
        mission = self.load_mission()

        return satellites, targets, mission