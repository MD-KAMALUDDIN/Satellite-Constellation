from datetime import datetime, timedelta, timezone

import pytest

from satellite_optimizer.models.mission import (
    ExecutionConfig,
    Mission,
    MissionConstraints,
    MissionScoringWeights,
    QuantumConfig,
    SimulationConfig,
)
from satellite_optimizer.models.observation import (
    ObservationOpportunity,
)
from satellite_optimizer.models.target import Target
from satellite_optimizer.optimization.reduction import (
    reduce_candidates,
    score_opportunity,
)


def make_mission() -> Mission:
    return Mission(
        name="Test Mission",
        simulation=SimulationConfig(
            duration_hours=24.0,
            time_step_seconds=30.0,
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
            max_candidates=2,
        ),
        execution=ExecutionConfig(
            default_backend="local",
            allow_ibm_simulator=True,
            allow_ibm_hardware=True,
        ),
    )


def make_target(
    target_id: str = "TGT-01",
    priority: float = 10.0,
) -> Target:
    return Target(
        id=target_id,
        name="Test Target",
        latitude_deg=17.385,
        longitude_deg=78.4867,
        priority=priority,
        observation_duration_sec=10.0,
        data_size_mb=500.0,
        deadline_hours=24.0,
    )


def make_opportunity(
    seconds_after_start: int = 60,
    image_quality: float = 0.8,
    target_id: str = "TGT-01",
) -> ObservationOpportunity:
    start_time = (
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )
        + timedelta(seconds=seconds_after_start)
    )

    return ObservationOpportunity(
        satellite_id="SAT-01",
        target_id=target_id,
        start_time=start_time,
        end_time=start_time + timedelta(seconds=10),
        elevation_deg=70.0,
        off_nadir_deg=10.0,
        slew_angle_deg=10.0,
        slew_time_sec=5.0,
        image_quality=image_quality,
        energy_cost_wh=0.3,
        data_size_mb=500.0,
        feasible=True,
    )


def test_score_opportunity_is_normalized():
    mission = make_mission()
    target = make_target()

    score = score_opportunity(
        opportunity=make_opportunity(),
        target=target,
        mission=mission,
        mission_start=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert 0.0 < score <= 1.0


def test_higher_image_quality_improves_score():
    mission = make_mission()
    target = make_target()

    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    low_quality = score_opportunity(
        opportunity=make_opportunity(
            image_quality=0.2,
        ),
        target=target,
        mission=mission,
        mission_start=mission_start,
    )

    high_quality = score_opportunity(
        opportunity=make_opportunity(
            image_quality=0.9,
        ),
        target=target,
        mission=mission,
        mission_start=mission_start,
    )

    assert high_quality > low_quality


def test_reduction_respects_max_candidates():
    mission = make_mission()

    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    opportunities = [
        make_opportunity(
            seconds_after_start=i * 60,
        )
        for i in range(10)
    ]

    candidates = reduce_candidates(
        opportunities=opportunities,
        targets=[make_target()],
        mission=mission,
        mission_start=mission_start,
    )

    assert len(candidates) == 2


def test_candidates_are_sorted_best_first():
    mission = make_mission()

    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    opportunities = [
        make_opportunity(
            seconds_after_start=60,
            image_quality=0.2,
        ),
        make_opportunity(
            seconds_after_start=120,
            image_quality=0.9,
        ),
    ]

    candidates = reduce_candidates(
        opportunities=opportunities,
        targets=[make_target()],
        mission=mission,
        mission_start=mission_start,
    )

    assert len(candidates) == 2
    assert candidates[0].score >= candidates[1].score


def test_infeasible_opportunities_are_removed():
    mission = make_mission()

    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    feasible = make_opportunity()

    infeasible = ObservationOpportunity(
        satellite_id="SAT-02",
        target_id="TGT-01",
        start_time=mission_start,
        end_time=mission_start + timedelta(seconds=10),
        elevation_deg=5.0,
        off_nadir_deg=40.0,
        slew_angle_deg=40.0,
        slew_time_sec=20.0,
        image_quality=0.1,
        energy_cost_wh=0.3,
        data_size_mb=500.0,
        feasible=False,
    )

    candidates = reduce_candidates(
        opportunities=[
            feasible,
            infeasible,
        ],
        targets=[make_target()],
        mission=mission,
        mission_start=mission_start,
    )

    assert len(candidates) == 1
    assert (
        candidates[0].opportunity.satellite_id
        == "SAT-01"
    )


def test_unknown_target_raises_error():
    mission = make_mission()

    mission_start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    unknown_target_opportunity = make_opportunity(
        target_id="UNKNOWN",
    )

    with pytest.raises(ValueError):
        from satellite_optimizer.optimization.reduction import (
            score_opportunities,
        )

        score_opportunities(
            opportunities=[
                unknown_target_opportunity,
            ],
            targets=[
                make_target(),
            ],
            mission=mission,
            mission_start=mission_start,
        )