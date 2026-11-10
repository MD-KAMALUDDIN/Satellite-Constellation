# Satellite Quantum Optimizer — v2

A hybrid classical-quantum framework for optimizing satellite observation scheduling. It combines orbital mechanics, constraint modeling, QUBO formulation, and QAOA execution to maximize mission value across multi-satellite constellations.

---

## Overview

Traditional satellite scheduling is NP-hard. This project encodes the scheduling problem as a Quadratic Unconstrained Binary Optimization (QUBO) and solves it using the Quantum Approximate Optimization Algorithm (QAOA) via Qiskit, with a classical fallback using Google OR-Tools.

---

## Architecture

```
satellite-quantum-optimizer/
├── src/satellite_optimizer/
│   ├── models/          # Mission, satellite, target, sensor, observation
│   ├── physics/         # Orbit propagation, visibility, geometry, image quality
│   ├── constraints/     # Battery, slew rate, storage, visibility validation
│   ├── scheduling/      # Opportunity detection, candidate generation, scoring
│   ├── optimization/    # QUBO formulation, classical solver, objective, reduction
│   ├── quantum/         # QAOA circuit, Qiskit backends (IBM + local), metrics
│   ├── hybrid/          # Classical-quantum feedback loop, evaluator
│   ├── simulation/      # Scenario generation
│   ├── api/             # Decision API endpoint
│   └── utils/           # Config loader, logging
├── config/              # mission.yaml, satellites.yaml, targets.yaml
├── dashboard/           # Streamlit app + Plotly components
├── notebooks/           # Exploration and analysis notebooks
├── scripts/             # CLI runners for classical, QAOA, hybrid
├── tests/               # Full test suite
```

---

## Key Capabilities

- **Orbital propagation** using Skyfield with SGP4
- **Visibility windows** computed from ground target geometry
- **Constraint enforcement**: battery budget, slew angle limits, onboard storage
- **QUBO encoding** of the scheduling problem with penalty terms
- **QAOA solver** with configurable depth (p-layers) on Qiskit Aer or IBM Quantum
- **Classical solver** via OR-Tools CP-SAT for benchmarking
- **Hybrid loop** that iterates between classical pre-filtering and quantum optimization
- **Interactive dashboard** built with Streamlit and Plotly (3D globe, circuit view, results panel)

---

## Installation

Requires Python 3.12.

```bash
git clone https://github.com/your-org/satellite-quantum-optimizer.git
cd satellite-quantum-optimizer
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

---

## Configuration

Edit the YAML files in `config/` before running:

| File | Purpose |
|---|---|
| `mission.yaml` | Time window, optimization weights, QAOA depth |
| `satellites.yaml` | TLE data, sensor specs, battery/storage limits |
| `targets.yaml` | Ground targets, priority scores, coordinates |

---

## Usage

**Generate a scenario:**
```bash
python scripts/generate_scenario.py
```

**Run classical solver:**
```bash
python scripts/run_classical.py
```

**Run QAOA solver:**
```bash
python scripts/run_qaoa.py
```

**Run hybrid loop:**
```bash
python scripts/run_hybrid.py
```

**Launch dashboard:**
```bash
streamlit run dashboard/app.py
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_orbit_exploration.ipynb` | Visualize satellite orbits and ground tracks |
| `02_visibility_analysis.ipynb` | Compute and plot visibility windows |
| `03_classical_optimization.ipynb` | OR-Tools scheduling baseline |
| `04_qubo_analysis.ipynb` | QUBO matrix construction and analysis |
| `05_qaoa_analysis.ipynb` | QAOA circuit depth, convergence, and results |

---

## Testing

```bash
pytest tests/
```

Test coverage includes: orbit propagation, visibility, geometry, constraints, QUBO, QAOA pipeline, classical solver, hybrid loop, scoring, and config loading.

---

## Stack

| Layer | Technology |
|---|---|
| Quantum | Qiskit 2.x, Qiskit Aer, Qiskit Optimization, IBM Quantum Runtime |
| Classical | Google OR-Tools 9.x |
| Orbital mechanics | Skyfield 1.x |
| Data | NumPy, SciPy, Pandas |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Config & models | PyYAML, Pydantic v2 |

---

## IBM Quantum Backend

To run on real quantum hardware, set your IBM Quantum credentials:

```bash
export IBM_QUANTUM_TOKEN=<your_token>
```

Or configure via `qiskit-ibm-runtime` directly. The backend selection is handled in `src/satellite_optimizer/quantum/ibm_backend.py`.

---

## License

See [LICENSE](LICENSE).
