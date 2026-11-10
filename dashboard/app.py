from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
HTML_FILE = Path(__file__).parent / "index.html"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satellite_optimizer.hybrid.evaluator import run_hybrid_evaluation  # noqa: E402
from satellite_optimizer.optimization.reduction import reduce_candidates  # noqa: E402
from satellite_optimizer.quantum.circuit import build_bound_circuit  # noqa: E402
from satellite_optimizer.physics.geometry import eci_to_ecef  # noqa: E402
from satellite_optimizer.physics.orbit import CircularOrbit  # noqa: E402
EARTH_RADIUS_KM = 6378.137
from satellite_optimizer.physics.propagation import propagate_circular_orbit  # noqa: E402
from satellite_optimizer.scheduling.opportunities import generate_opportunities  # noqa: E402
from satellite_optimizer.utils.config import ConfigLoader  # noqa: E402

app = FastAPI()

_config_cache = None
_EPOCH_S = time.time()  # server start = t=0 for orbit propagation


def get_config():
    global _config_cache
    if _config_cache is None:
        config_dir = PROJECT_ROOT / "config"
        _config_cache = ConfigLoader(config_dir=config_dir).load_all()
    return _config_cache


def _sat_geo(sat, elapsed_s: float) -> dict:
    """Return lat/lon/alt/velocity for a satellite at elapsed_s from epoch."""
    orbit = CircularOrbit(
        altitude_km=sat.altitude_km,
        inclination_deg=sat.inclination_deg,
        raan_deg=sat.raan_deg,
        phase_deg=sat.phase_deg,
    )
    eci = propagate_circular_orbit(orbit, elapsed_s)
    ecef = eci_to_ecef(eci.x_km, eci.y_km, eci.z_km, elapsed_s)

    r = math.sqrt(ecef.x_km**2 + ecef.y_km**2 + ecef.z_km**2)
    lat = math.degrees(math.asin(ecef.z_km / r))
    lon = math.degrees(math.atan2(ecef.y_km, ecef.x_km))
    alt = r - EARTH_RADIUS_KM

    # velocity via finite difference
    eci2 = propagate_circular_orbit(orbit, elapsed_s + 1.0)
    vel = math.sqrt(
        (eci2.x_km - eci.x_km) ** 2
        + (eci2.y_km - eci.y_km) ** 2
        + (eci2.z_km - eci.z_km) ** 2
    )
    return {"lat": lat, "lon": lon, "alt": alt, "vel_km_s": vel}


def _ground_track(sat, elapsed_s: float, minutes: int = 90) -> list:
    orbit = CircularOrbit(
        altitude_km=sat.altitude_km,
        inclination_deg=sat.inclination_deg,
        raan_deg=sat.raan_deg,
        phase_deg=sat.phase_deg,
    )
    track = []
    for dt in range(-minutes // 2, minutes // 2 + 1, 2):
        t = elapsed_s + dt * 60
        eci = propagate_circular_orbit(orbit, t)
        ecef = eci_to_ecef(eci.x_km, eci.y_km, eci.z_km, t)
        r = math.sqrt(ecef.x_km**2 + ecef.y_km**2 + ecef.z_km**2)
        track.append({
            "lat": math.degrees(math.asin(ecef.z_km / r)),
            "lon": math.degrees(math.atan2(ecef.y_km, ecef.x_km)),
        })
    return track


@app.get("/", response_class=HTMLResponse)
async def index():
    content = HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def get_config_endpoint():
    satellites, targets, mission = get_config()
    return JSONResponse({
        "satellites": [
            {
                "id": s.id,
                "name": s.name,
                "altitude_km": s.altitude_km,
                "inclination_deg": s.inclination_deg,
                "raan_deg": s.raan_deg,
                "phase_deg": s.phase_deg,
                "battery_capacity_wh": s.battery_capacity_wh,
                "initial_battery_wh": s.initial_battery_wh,
                "storage_capacity_gb": s.storage_capacity_gb,
                "initial_storage_gb": s.initial_storage_gb,
                "slew_rate_deg_per_sec": s.slew_rate_deg_per_sec,
            }
            for s in satellites
        ],
        "targets": [
            {
                "id": t.id,
                "name": t.name,
                "lat": t.latitude_deg,
                "lon": t.longitude_deg,
                "priority": t.priority,
                "observation_duration_sec": t.observation_duration_sec,
                "data_size_mb": t.data_size_mb,
                "deadline_hours": t.deadline_hours,
            }
            for t in targets
        ],
        "mission": {
            "name": mission.name,
            "duration_hours": mission.simulation.duration_hours,
            "time_step_seconds": mission.simulation.time_step_seconds,
            "min_elevation_deg": mission.constraints.minimum_elevation_deg,
            "min_battery_reserve_pct": mission.constraints.minimum_battery_reserve_percent,
            "max_off_nadir_deg": mission.constraints.maximum_off_nadir_deg,
            "qaoa_depth": mission.quantum.default_depth,
            "shots": mission.quantum.shots,
            "max_candidates": mission.quantum.max_candidates,
            "scoring": {
                "task_value_weight": mission.scoring.task_value_weight,
                "image_quality_weight": mission.scoring.image_quality_weight,
                "future_value_weight": mission.scoring.future_value_weight,
                "energy_weight": mission.scoring.energy_weight,
                "slew_weight": mission.scoring.slew_weight,
                "data_weight": mission.scoring.data_weight,
            },
        },
    })


@app.get("/api/satellites/positions")
async def satellite_positions():
    satellites, _, _ = get_config()
    elapsed_s = time.time() - _EPOCH_S
    return JSONResponse({
        "elapsed_s": elapsed_s,
        "positions": [
            {
                "id": s.id,
                "name": s.name,
                **_sat_geo(s, elapsed_s),
                "battery_wh": s.initial_battery_wh,
                "battery_pct": round(s.initial_battery_wh / s.battery_capacity_wh * 100, 1),
                "storage_used_gb": s.initial_storage_gb,
                "storage_pct": round(s.initial_storage_gb / s.storage_capacity_gb * 100, 1),
                "slew_rate": s.slew_rate_deg_per_sec,
                "period_min": round(
                    2 * math.pi * math.sqrt(
                        (EARTH_RADIUS_KM + s.altitude_km) ** 3 / 398600.4418
                    ) / 60, 2
                ),
            }
            for s in satellites
        ],
    })


@app.get("/api/satellites/{sat_id}/track")
async def satellite_track(sat_id: str):
    satellites, _, _ = get_config()
    sat = next((s for s in satellites if s.id == sat_id), None)
    if sat is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    elapsed_s = time.time() - _EPOCH_S
    return JSONResponse({"id": sat_id, "track": _ground_track(sat, elapsed_s)})


@app.get("/api/windows")
async def access_windows():
    satellites, targets, mission = get_config()
    mission_start = datetime.now(timezone.utc)
    opportunities = generate_opportunities(
        satellites=satellites,
        targets=targets,
        mission=mission,
        start_time=mission_start,
    )
    windows = [
        {
            "satellite_id": o.satellite_id,
            "target_id": o.target_id,
            "start": o.start_time.isoformat(),
            "end": o.end_time.isoformat(),
            "elevation_deg": round(o.elevation_deg, 2),
            "off_nadir_deg": round(o.off_nadir_deg, 2),
            "image_quality": round(o.image_quality, 3),
            "energy_cost_wh": round(o.energy_cost_wh, 3),
        }
        for o in opportunities
    ]
    return JSONResponse({"count": len(windows), "windows": windows})


_last_qaoa_circuit_svg: str | None = None
_last_qaoa_meta: dict | None = None
_last_qubo: dict | None = None


def _render_circuit_svg(qc) -> str:
    """Render a Qiskit QuantumCircuit to a readable SVG.

    Produces a single-row layout (fold=-1), then rewrites the root
    SVG width/height from pt units to px so browsers render it at a
    sensible size inside a horizontally-scrollable container.
    Target: 300 px tall; width scales proportionally from the viewBox.
    """
    import io, re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qiskit.visualization import circuit_drawer

    style = {
        "backgroundcolor": "#020a12",
        "textcolor": "#d8ecf7",
        "gatefacecolor": "#0a2030",
        "gatetextcolor": "#16d9ff",
        "barrierfacecolor": "#1d3a52",
        "arrowcolor": "#16d9ff",
        "linecolor": "#7aa3b8",
        "creglinecolor": "#39ffb0",
        "displaycolor": {
            "h":       ["#16d9ff", "#020a12"],
            "cx":      ["#ff3d9a", "#020a12"],
            "rz":      ["#b47cff", "#020a12"],
            "rx":      ["#39ffb0", "#020a12"],
            "rzz":     ["#ffb454", "#020a12"],
            "measure": ["#7aa3b8", "#020a12"],
        },
    }

    fig = circuit_drawer(qc, output="mpl", fold=-1, scale=0.5, style=style)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=72)
    plt.close(fig)
    buf.seek(0)
    svg = buf.read().decode("utf-8")

    # The SVG root has width/height in pt (e.g. "7098pt x 510pt").
    # Keep the viewBox intact and replace width/height with px values
    # scaled so height == 300px, making it horizontally scrollable.
    m = re.search(
        r'<svg([^>]*)\swidth="([^"]+)pt"([^>]*)\sheight="([^"]+)pt"',
        svg,
    )
    if m:
        vb_w = float(m.group(2))
        vb_h = float(m.group(4))
        target_h = 300
        target_w = round(vb_w / vb_h * target_h)
        svg = re.sub(
            r'(<svg[^>]*)\swidth="[^"]+pt"([^>]*)\sheight="[^"]+pt"',
            rf'\1 width="{target_w}px"\2 height="{target_h}px"',
            svg,
            count=1,
        )

    return svg


class OptimizeRequest(BaseModel):
    target_id: str | None = None


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    import traceback
    try:
        return await _run_optimize(req)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _run_optimize(req: OptimizeRequest):
    satellites, targets, mission = get_config()
    mission_start = datetime.now(timezone.utc)

    opportunities = generate_opportunities(
        satellites=satellites,
        targets=targets,
        mission=mission,
        start_time=mission_start,
    )

    if req.target_id:
        opportunities = [o for o in opportunities if o.target_id == req.target_id]

    if not opportunities:
        return JSONResponse({"error": "no opportunities found"}, status_code=400)

    candidates = reduce_candidates(
        opportunities=opportunities,
        targets=targets,
        mission=mission,
        mission_start=mission_start,
    )

    result = run_hybrid_evaluation(
        candidates=candidates,
        satellites=satellites,
        qaoa_depth=mission.quantum.default_depth,
        shots=mission.quantum.shots,
        max_iterations=mission.quantum.max_iterations,
        seed=42,
        classical_time_limit_seconds=30.0,
    )

    global _last_qaoa_circuit_svg, _last_qaoa_meta
    qaoa = result.qaoa_result
    classical = result.classical_result
    validation = result.qaoa_validation
    comparison = result.comparison

    try:
        d = qaoa.circuit.depth
        final_qc = build_bound_circuit(
            qaoa.circuit,
            qaoa.best_parameters[:d].tolist(),
            qaoa.best_parameters[d:].tolist(),
        )
        _last_qaoa_circuit_svg = _render_circuit_svg(final_qc)
    except Exception:
        _last_qaoa_circuit_svg = None

    global _last_qubo
    _last_qubo = {
        "matrix": result.qubo.matrix.tolist(),
        "labels": [
            f"{c.opportunity.satellite_id}\u2192{c.opportunity.target_id}"
            for c in candidates
        ],
        "conflict_penalty": result.qubo.conflict_penalty,
        "energy_penalty": result.qubo.energy_penalty,
        "n_observation": result.qubo.num_observation_variables,
        "n_slack": result.qubo.num_slack_variables,
        "n": result.qubo.num_variables,
    }

    _last_qaoa_meta = {
        "n_observation_variables": result.qubo.num_observation_variables,
        "n_slack_variables": result.qubo.num_slack_variables,
        "n_total_qubits": result.qubo.num_qubits,
        "qaoa_depth": qaoa.circuit.depth,
        "circuit_depth": qaoa.circuit_depth,
        "gate_count": qaoa.gate_count,
        "two_qubit_gates": qaoa.two_qubit_gate_count,
        "shots": qaoa.shots,
        "backend": qaoa.backend_name,
    }

    selected = [
        {
            "satellite_id": c.opportunity.satellite_id,
            "target_id": c.opportunity.target_id,
            "start_time": c.opportunity.start_time.isoformat(),
            "end_time": c.opportunity.end_time.isoformat(),
            "elevation_deg": round(c.opportunity.elevation_deg, 2),
            "off_nadir_deg": round(c.opportunity.off_nadir_deg, 2),
            "image_quality": round(c.opportunity.image_quality, 3),
            "energy_cost_wh": round(c.opportunity.energy_cost_wh, 3),
            "score": round(c.score, 6),
        }
        for c in sorted(
            result.final_selected_candidates,
            key=lambda c: c.opportunity.start_time,
        )
    ]

    return JSONResponse({
        "final_method": result.final_method,
        "selected": selected,
        "qaoa": {
            "n_observation_variables": result.qubo.num_observation_variables,
            "n_slack_variables": result.qubo.num_slack_variables,
            "n_total_qubits": result.qubo.num_qubits,
            "depth": qaoa.circuit.depth,
            "circuit_depth": qaoa.circuit_depth,
            "gate_count": qaoa.gate_count,
            "two_qubit_gates": qaoa.two_qubit_gate_count,
            "shots": qaoa.shots,
            "backend": qaoa.backend_name,
            "raw_best_bitstring": qaoa.raw_best_bitstring,
            "raw_best_energy": round(qaoa.raw_best_energy, 6),
            "expectation_energy": round(qaoa.best_energy, 6),
            "feasible_sample_count": validation.feasible_sample_count,
            "total_unique_samples": len(qaoa.counts),
            "feasible": validation.feasible,
            "best_feasible_score": round(comparison.qaoa_score, 6) if comparison.qaoa_score is not None else None,
            "violations": validation.violations,
        },
        "classical": {
            "status": classical.solver_status,
            "n_selected": len(classical.selected_candidates),
            "objective": round(comparison.classical_score, 6),
            "feasible": True,
        },
        "comparison": {
            "classical_score": round(comparison.classical_score, 6),
            "qaoa_score": round(comparison.qaoa_score, 6) if comparison.qaoa_score is not None else None,
            "score_gap": round(comparison.score_gap, 6) if comparison.score_gap is not None else None,
            "relative_gap_pct": round(comparison.qaoa_relative_gap_percent, 2) if comparison.qaoa_relative_gap_percent is not None else None,
            "same_solution": comparison.same_solution,
        },
    })


@app.get("/api/qubo")
async def get_qubo():
    if _last_qubo is None:
        return JSONResponse({"error": "Run optimization first."}, status_code=404)
    return JSONResponse(_last_qubo)


@app.get("/api/circuit")
async def get_circuit():
    if _last_qaoa_circuit_svg is None:
        return Response(
            content="Circuit not available. Run optimization first.",
            media_type="text/plain",
            status_code=404,
        )
    return Response(
        content=_last_qaoa_circuit_svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8501, reload=True)
