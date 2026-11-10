from datetime import datetime, timezone

from satellite_optimizer.models.mission import (
    Mission,
    MissionConstraints,
    MissionScoringWeights,
    QuantumConfig,
    SimulationConfig,
    ExecutionConfig,
)
from satellite_optimizer.models.satellite import Satellite
from satellite_optimizer.models.target import Target
from satellite_optimizer.scheduling.opportunities import (
    calculate_energy_cost,
    calculate_image_quality,
    calculate_slew_time,
    generate_opportunities,
)


def make_mission() -> Mission:
    return Mission(
        name="Test Mission",
        simulation=SimulationConfig(
            duration_hours=1,
            time_step_seconds=30,
        ),
        constraints=MissionConstraints(
            minimum_elevation_deg=10.0,
            minimum_battery_reserve_percent=20.0,
            maximum_off_nadir_deg=30.0,
        ),
        scoring=MissionScoringWeights(
            task_value_weight=0.40,
            image_quality_weight=0.20,
            future_value_weight=0.15,
            energy_weight=0.10,
            slew_weight=0.10,
            data_weight=0.05,
        ),
        quantum=QuantumConfig(
            algorithm="QAOA",
            default_depth=2,
            shots=2048,
            max_candidates=20,
        ),
        execution=ExecutionConfig(
            default_backend="local",
            allow_ibm_simulator=True,
            allow_ibm_hardware=True,
        ),
    )


def make_satellite() -> Satellite:
    return Satellite(
        id="SAT-TEST",
        name="Test Satellite",
        altitude_km=550.0,
        inclination_deg=53.0,
        raan_deg=0.0,
        phase_deg=0.0,
        battery_capacity_wh=1000.0,
        initial_battery_wh=850.0,
        storage_capacity_gb=256.0,
        initial_storage_gb=40.0,
        slew_rate_deg_per_sec=2.0,
    )


def make_target() -> Target:
    return Target(
        id="TGT-TEST",
        name="Test Target",
        latitude_deg=0.0,
        longitude_deg=0.0,
        priority=10,
        observation_duration_sec=10,
        data_size_mb=500,
        deadline_hours=24,
    )


def test_energy_cost():
    assert calculate_energy_cost(10.0) == 100.0 / 360.0


def test_slew_time():
    assert calculate_slew_time(
        slew_angle_deg=20.0,
        slew_rate_deg_per_sec=2.0,
    ) == 10.0


def test_image_quality_is_normalized():
    quality = calculate_image_quality(
        elevation_deg=90.0,
        off_nadir_deg=0.0,
    )

    assert 0.0 <= quality <= 1.0
    assert quality == 1.0


def test_generate_opportunities():
    opportunities = generate_opportunities(
        satellites=[make_satellite()],
        targets=[make_target()],
        mission=make_mission(),
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert len(opportunities) > 0

    opportunity = opportunities[0]

    assert opportunity.satellite_id == "SAT-TEST"
    assert opportunity.target_id == "TGT-TEST"
    assert opportunity.feasible is True
    assert opportunity.end_time > opportunity.start_time
    assert opportunity.image_quality > 0.0