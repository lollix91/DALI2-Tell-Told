# DALI2-Tell-Told: Experimental Evaluation of Declarative Communication Filters

Companion experiment for the paper:

> **Declarative Communication Filters for Controlled LLM Integration
> in Logic-Based Multi-Agent Systems**
> Lorenzo De Lauretis, Stefania Costantini
> RCRA 2026 @ FLoC 2026, Lisbon, Portugal

This repository evaluates DALI2's **tell/told filtering mechanism** applied to
LLM (AI Oracle) interactions. The experiment measures filter correctness,
LLM response parseability, state-dependent accuracy, and computational overhead
across three scenarios using GPT-4o via OpenRouter.

---

## What This Experiment Tests

DALI2 agents declare **tell rules** (which queries may be sent to the AI Oracle)
and **told rules** (which AI responses will be accepted), with optional
state-dependent body conditions and priority ordering. The experiment validates that:

1. **Tell filter** — off-domain queries are blocked *before* any LLM call is made.
2. **Told filter** — LLM responses not matching declared patterns are rejected.
3. **State-dependent conditions** — the same response is accepted or rejected
   depending on the agent's current belief state.
4. **Overhead** — filter evaluation is computationally negligible (~0 ms) compared
   to LLM latency (~1,800 ms).

---

## Contents

| File | Description |
|------|-------------|
| `experiment.py` | Python simulation: replicates DALI2's filter algorithm, calls GPT-4o |
| `experiment_agents.pl` | DALI2 agent definitions for end-to-end testing via Docker |
| `results.json` | Raw results and statistics from the reported experiment run |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

### For the Python simulation (no DALI2 needed)

- Python 3.8+
- `pip install -r requirements.txt`
- An [OpenRouter](https://openrouter.ai/) API key with access to `openai/gpt-4o`

### For end-to-end testing with DALI2

- All of the above
- [DALI2](https://github.com/AAAI-DISIM-UnivAQ/DALI2) cloned locally
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Quick Start

### Step 1 — Set Your API Key

The API key is **never hardcoded**. Set it as an environment variable before
starting DALI2 and before running the script:

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

**Linux / macOS:**
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

If the variable is not set, the script will prompt for it at runtime
(input is hidden via `getpass`).

### Step 2 — Start DALI2

Copy `experiment_agents.pl` to your DALI2 clone's `examples/` directory, then:

```powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:AGENT_FILE = "examples/experiment_agents.pl"
cd path\to\DALI2
docker compose up --build
```

```bash
# Linux / macOS
OPENROUTER_API_KEY="sk-or-..." \
AGENT_FILE=examples/experiment_agents.pl \
docker compose up --build
```

Open **http://localhost:8080** to see the web dashboard.

### Step 3 — Run the Experiment

```bash
pip install -r requirements.txt
python experiment.py
```

The script automatically:
1. Injects `run_test(Id, Context)` events into each agent via `/api/inject`
2. Sets agent state (`set_status(active/idle)`) as required per test case
3. Polls `/api/beliefs` until `test_result(Id, Outcome)` appears
4. Records outcome (`blocked` / `rejected` / `accepted`) and latency
5. Saves all results and statistics to `results.json`

Optional arguments:
```bash
python experiment.py --url http://localhost:8080 --model openai/gpt-4o --repetitions 3
```

### Manual Testing via REST API

#### Test the tell filter (blocked immediately — no LLM call):
```powershell
curl.exe -X POST http://localhost:8080/api/inject `
  -H "Content-Type: application/json" `
  -d '{"agent":"crop_advisor","event":"run_test(99,market_price(wheat))"}'

# Check result (should show: test_result(99,blocked))
curl.exe "http://localhost:8080/api/beliefs?agent=crop_advisor"
```

#### Test the told filter (LLM called, response may be rejected):
```powershell
curl.exe -X POST http://localhost:8080/api/inject `
  -H "Content-Type: application/json" `
  -d '{"agent":"crop_advisor","event":"run_test(100,soil_analysis(moisture(25),ph(6.5),field(north)))"}'

curl.exe "http://localhost:8080/api/beliefs?agent=crop_advisor"
```

#### Test state-dependent filtering:
```powershell
# Set agent to idle then run test → response rejected
curl.exe -X POST http://localhost:8080/api/inject `
  -H "Content-Type: application/json" `
  -d '{"agent":"state_test","event":"set_status(idle)"}'

curl.exe -X POST http://localhost:8080/api/inject `
  -H "Content-Type: application/json" `
  -d '{"agent":"state_test","event":"run_test(101,suggestion_request(optimize_irrigation))"}'

curl.exe "http://localhost:8080/api/beliefs?agent=state_test"
```

### How Test Results Are Stored

Results are stored as `test_result(Id, Outcome)` beliefs, where `Outcome` is:
- `accepted` — LLM called, response matched told filter
- `blocked` — blocked by tell filter (no LLM call)
- `rejected` — LLM called, response rejected by told filter

Query them:
```
GET http://localhost:8080/api/beliefs?agent=crop_advisor
```

---

## Experiment Results

The reported results (in `results.json`) were obtained with GPT-4o, temperature 0.3,
max 100 tokens, 3 repetitions per test case (60 executions total):

| Scenario       | Tests | Blocked | Accepted | Rejected | Accuracy | Avg latency |
|----------------|-------|---------|----------|----------|----------|-------------|
| Agriculture    | 21    | 9 (43%) | 12 (57%) | 0        | 100%     | 1,718 ms    |
| Emergency      | 21    | 9 (43%) | 0        | 12 (57%) | 100%     | 1,976 ms    |
| State-dependent| 18    | 6 (33%) | 6 (33%)  | 6 (33%)  | 100%     | 1,840 ms    |
| **Total**      | **60**| **24**  | **18**   | **18**   | **100%** | **1,845 ms**|

Filter overhead was below 0.1 ms in all cases (ratio ~10⁻⁴ vs LLM latency).

---

## Scenarios

### Scenario A — Smart Agriculture (`crop_advisor`)

Tell rules allow only `soil_analysis/3` and `weather_analysis/3`.
Told rules accept `suggestion/1` (priority 80) and `recommendation/2`
(priority 90, only when `status(active)`). Off-domain queries (e.g.,
`market_price/1`) are blocked.

### Scenario B — Emergency Response (`coordinator`)

Tell rules allow `analyze/1` and `log_event/3`.
Told rules establish a priority queue: `emergency/2` (200) > `alert/2` (100)
> `sensor_data/1` (30, active only) > `calibration_request` (10).
All 12 LLM responses were rejected in experiments because GPT-4o returned
`suggestion/action` terms not in the whitelist — demonstrating fail-safe
protection against LLM output drift.

### Scenario C — State-Dependent Filtering (`state_test`)

Tell rules allow `suggestion_request/1`.
Told rules only accept responses when `status(active)`.
The same query accepted with `active` is rejected with `idle`,
demonstrating state-dependent body condition evaluation.

---

## Citation

```bibtex
@inproceedings{delauretis2026telltold,
  author    = {Lorenzo De Lauretis and Stefania Costantini},
  title     = {Declarative Communication Filters for Controlled {LLM} Integration
               in Logic-Based Multi-Agent Systems},
  booktitle = {Proceedings of the 33rd International Workshop on Experimental
               Evaluation of Algorithms for Solving Problems with Combinatorial
               Explosion ({RCRA} 2026)},
  year      = {2026}
}
```
