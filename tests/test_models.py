from satellite_optimizer.models.satellite import Satellite
from satellite_optimizer.models.target import Target


def test_satellite_creation():
    satellite = Satellite(
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

    assert satellite.id == "SAT-TEST"
    assert satellite.altitude_km == 550.0
    assert satellite.initial_battery_wh == 850.0


def test_target_creation():
    target = Target(
        id="TGT-TEST",
        name="Test Target",
        latitude_deg=17.385,
        longitude_deg=78.4867,
        priority=10,
        observation_duration_sec=10,
        data_size_mb=500,
        deadline_hours=24,
    )

    assert target.id == "TGT-TEST"
    assert target.name == "Test Target"
    assert target.priority == 10