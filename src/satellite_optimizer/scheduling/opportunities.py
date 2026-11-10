from datetime import datetime, timedelta, timezone

from satellite_optimizer.models.mission import Mission
from satellite_optimizer.models.observation import ObservationOpportunity
from satellite_optimizer.models.satellite import Satellite
from satellite_optimizer.models.target import Target
from satellite_optimizer.physics.geometry import create_ground_point
from satellite_optimizer.physics.orbit import CircularOrbit
from satellite_optimizer.physics.visibility import check_visibility


def calculate_image_quality(
    elevation_deg: float,
    off_nadir_deg: float,
) -> float:
    """
    Estimate image quality from observation geometry.

    V1 uses a normalized heuristic:
    - higher elevation improves quality
    - lower off-nadir angle improves quality

    Returns a value between 0 and 1.
    """

    elevation_score = max(
        0.0,
        min(1.0, elevation_deg / 90.0),
    )

    off_nadir_score = max(
        0.0,
        1.0 - off_nadir_deg / 30.0,
    )

    return (
        0.6 * elevation_score
        + 0.4 * off_nadir_score
    )


def calculate_energy_cost(
    observation_duration_sec: float,
) -> float:
    """
    V1 placeholder for payload energy consumption.

    Sensor power will become configurable when the sensor
    model is connected to the mission configuration.
    """

    observation_power_w = 100.0

    return (
        observation_power_w
        * observation_duration_sec
        / 3600.0
    )


def calculate_slew_time(
    slew_angle_deg: float,
    slew_rate_deg_per_sec: float,
) -> float:
    if slew_rate_deg_per_sec <= 0.0:
        raise ValueError(
            "Slew rate must be greater than zero."
        )

    return slew_angle_deg / slew_rate_deg_per_sec


def generate_opportunities(
    satellites: list[Satellite],
    targets: list[Target],
    mission: Mission,
    start_time: datetime,
) -> list[ObservationOpportunity]:
    """
    Scan the mission timeline and generate feasible
    observation opportunities.

    The scanner evaluates each satellite-target pair
    at the configured time step.
    """

    opportunities: list[ObservationOpportunity] = []

    duration_seconds = (
        mission.simulation.duration_hours * 3600.0
    )

    time_step = mission.simulation.time_step_seconds

    for satellite in satellites:
        orbit = CircularOrbit(
            altitude_km=satellite.altitude_km,
            inclination_deg=satellite.inclination_deg,
            raan_deg=satellite.raan_deg,
            phase_deg=satellite.phase_deg,
        )

        for target in targets:
            ground_point = create_ground_point(
                latitude_deg=target.latitude_deg,
                longitude_deg=target.longitude_deg,
            )

            elapsed = 0.0

            while elapsed < duration_seconds:
                visibility = check_visibility(
                    orbit=orbit,
                    target=ground_point,
                    elapsed_seconds=elapsed,
                    constraints=mission.constraints,
                )

                if visibility.visible:
                    observation_duration = (
                        target.observation_duration_sec
                    )

                    start_offset = timedelta(
                        seconds=elapsed
                    )

                    observation_start = (
                        start_time + start_offset
                    )

                    observation_end = (
                        observation_start
                        + timedelta(
                            seconds=observation_duration
                        )
                    )

                    slew_angle = (
                        visibility.geometry.off_nadir_deg
                    )

                    slew_time = calculate_slew_time(
                        slew_angle_deg=slew_angle,
                        slew_rate_deg_per_sec=(
                            satellite.slew_rate_deg_per_sec
                        ),
                    )

                    image_quality = calculate_image_quality(
                        elevation_deg=(
                            visibility.geometry.elevation_deg
                        ),
                        off_nadir_deg=(
                            visibility.geometry.off_nadir_deg
                        ),
                    )

                    energy_cost = calculate_energy_cost(
                        observation_duration_sec=(
                            observation_duration
                        )
                    )

                    opportunities.append(
                        ObservationOpportunity(
                            satellite_id=satellite.id,
                            target_id=target.id,
                            start_time=observation_start,
                            end_time=observation_end,
                            elevation_deg=(
                                visibility.geometry.elevation_deg
                            ),
                            off_nadir_deg=(
                                visibility.geometry.off_nadir_deg
                            ),
                            slew_angle_deg=slew_angle,
                            slew_time_sec=slew_time,
                            image_quality=image_quality,
                            energy_cost_wh=energy_cost,
                            data_size_mb=target.data_size_mb,
                            feasible=True,
                        )
                    )

                elapsed += time_step

    return opportunities