from satellite_optimizer.utils.config import ConfigLoader


def test_load_satellites():
    loader = ConfigLoader()

    satellites = loader.load_satellites()

    assert len(satellites) == 10
    assert satellites[0].id == "SAT-01"


def test_load_targets():
    loader = ConfigLoader()

    targets = loader.load_targets()

    assert len(targets) == 20
    assert targets[0].id == "TGT-01"


def test_load_mission():
    loader = ConfigLoader()

    mission = loader.load_mission()

    assert mission.name == "Satellite Quantum Optimization V1"
    assert mission.quantum.algorithm == "QAOA"
    assert mission.quantum.default_depth == 2
    assert mission.execution.default_backend == "local"