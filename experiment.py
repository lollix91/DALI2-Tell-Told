"""
DALI2 Experimental Evaluation: Declarative Communication Filters for LLM Integration
=====================================================================================

Tests DALI2's tell/told filtering mechanism applied to AI Oracle interactions.
Requires DALI2 running via Docker with experiment_agents.pl loaded.

Scenarios:
  A â€” Smart Agriculture   (crop_advisor agent)
  B â€” Emergency Response  (coordinator agent)
  C â€” State-Dependent     (state_test agent)

Usage:
    python experiment.py [--url http://localhost:8080] [--model openai/gpt-4o]

Setup:
    1. Set OPENROUTER_API_KEY env variable (or enter when prompted)
    2. Copy experiment_agents.pl to DALI2/examples/
    3. Start DALI2:
         Windows: $env:AGENT_FILE='examples/experiment_agents.pl'; docker compose up --build
         Linux:   AGENT_FILE=examples/experiment_agents.pl docker compose up --build
    4. python experiment.py
"""

import json
import time
import re
import sys
import os
import getpass
import argparse
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import requests

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DALI2_URL   = "http://localhost:8080"
DEFAULT_MODEL       = "openai/gpt-4o"
NUM_REPETITIONS     = 3      # repeat each test case for statistical robustness
POLL_INTERVAL       = 0.5   # seconds between belief polls
POLL_TIMEOUT        = 45.0  # max seconds to wait for LLM result
BLOCKED_TIMEOUT     = 3.0   # max seconds to wait for blocked result (no LLM call)


# ============================================================
# DALI2 REST CLIENT
# ============================================================

def dali2_inject(url: str, agent: str, event: str) -> dict:
    """Inject an event directly into an agent's queue."""
    r = requests.post(f"{url}/api/inject",
                      json={"agent": agent, "event": event},
                      timeout=10)
    r.raise_for_status()
    return r.json()


def dali2_get_beliefs(url: str, agent: str) -> list:
    """Return list of belief strings for the given agent."""
    r = requests.get(f"{url}/api/beliefs", params={"agent": agent}, timeout=10)
    r.raise_for_status()
    return [item["belief"] for item in r.json().get("beliefs", [])]


def dali2_set_ai_key(url: str, key: str):
    requests.post(f"{url}/api/ai/key", json={"key": key}, timeout=10).raise_for_status()


def dali2_set_ai_model(url: str, model: str):
    requests.post(f"{url}/api/ai/model", json={"model": model}, timeout=10).raise_for_status()


def dali2_ai_enabled(url: str) -> bool:
    r = requests.get(f"{url}/api/ai/status", timeout=10)
    return r.json().get("enabled", False)


def dali2_alive(url: str) -> bool:
    try:
        return requests.get(f"{url}/api/status", timeout=5).status_code == 200
    except Exception:
        return False


def poll_for_result(url: str, agent: str, test_id: int,
                    timeout: float) -> tuple:
    """
    Poll agent beliefs until test_result(test_id, Outcome) appears.
    Returns (outcome_str, elapsed_ms) or (None, elapsed_ms) on timeout.
    """
    pattern = re.compile(rf"^test_result\({re.escape(str(test_id))},(\w+)\)$")
    t0 = time.time()
    while time.time() - t0 < timeout:
        for b in dali2_get_beliefs(url, agent):
            m = pattern.match(b)
            if m:
                return m.group(1), (time.time() - t0) * 1000
        time.sleep(POLL_INTERVAL)
    return None, (time.time() - t0) * 1000




# ============================================================
# TEST SCENARIOS
# ============================================================

# Each entry: (scenario_name, agent, context, expected, state_override)
# state_override is None or "active" / "idle" â€” passed as set_status(state) event.
# expected is "blocked", "rejected", or "accepted_or_rejected".
TEST_CASES = [
    # --- Scenario A: Smart Agriculture (crop_advisor) ---
    # Tell filter: only soil_analysis/3 and weather_analysis/3 allowed.
    # Told filter: suggestion/1 (pri 80), recommendation/2 (pri 90, active only).
    ("agriculture", "crop_advisor",
     "soil_analysis(moisture(25),ph(6.5),field(north))",    "accepted_or_rejected", None),
    ("agriculture", "crop_advisor",
     "weather_analysis(temp(38),humidity(20),forecast(sunny))","accepted_or_rejected", None),
    ("agriculture", "crop_advisor",
     "market_price(wheat)",                                 "blocked",              None),
    ("agriculture", "crop_advisor",
     "financial_advice(invest,stocks)",                     "blocked",              None),
    ("agriculture", "crop_advisor",
     "generate_report(summary)",                            "blocked",              None),
    ("agriculture", "crop_advisor",
     "soil_analysis(moisture(60),ph(7.0),field(east))",     "accepted_or_rejected", "active"),
    ("agriculture", "crop_advisor",
     "soil_analysis(moisture(15),ph(5.5),field(south))",    "accepted_or_rejected", "idle"),

    # --- Scenario B: Emergency Response (coordinator) ---
    # Tell filter: only analyze/1 and log_event/3 allowed.
    # Told filter: emergency/2 (200), alert/2 (100), sensor_data/1 (30, active only), calibration_request (10).
    ("emergency",    "coordinator",
     "analyze(emergency(fire,building_a))",                 "accepted_or_rejected", None),
    ("emergency",    "coordinator",
     "analyze(situation(flood,district_5))",                "accepted_or_rejected", None),
    ("emergency",    "coordinator",
     "generate_report(summary)",                            "blocked",              None),
    ("emergency",    "coordinator",
     "send_email(admin,status_update)",                     "blocked",              None),
    ("emergency",    "coordinator",
     "query_database(personnel_records)",                   "blocked",              None),
    ("emergency",    "coordinator",
     "analyze(emergency(earthquake,downtown))",             "accepted_or_rejected", "active"),
    ("emergency",    "coordinator",
     "analyze(sensor_reading(temperature,85))",             "accepted_or_rejected", "idle"),

    # --- Scenario C: State-Dependent (state_test) ---
    # Tell filter: only suggestion_request/1 allowed.
    # Told filter: suggestion/1, recommendation/2 â€” both only when status(active).
    ("state_dependent", "state_test",
     "suggestion_request(optimize_irrigation_schedule)",    "accepted_or_rejected", "active"),
    ("state_dependent", "state_test",
     "suggestion_request(analyze_soil_conditions)",         "accepted_or_rejected", "active"),
    ("state_dependent", "state_test",
     "suggestion_request(optimize_irrigation_schedule)",    "rejected",             "idle"),
    ("state_dependent", "state_test",
     "suggestion_request(analyze_soil_conditions)",         "rejected",             "idle"),
    ("state_dependent", "state_test",
     "check_stock_market(tech_sector)",                     "blocked",              "active"),
    ("state_dependent", "state_test",
     "check_stock_market(tech_sector)",                     "blocked",              "idle"),
]


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

@dataclass
class Result:
    scenario:   str
    test_id:    int
    repetition: int
    agent:      str
    query:      str
    state:      Optional[str]
    expected:   str
    outcome:    str          # "blocked", "rejected", "accepted", "timeout", "error"
    correct:    bool
    latency_ms: float
    error:      str = ""


def run_experiment(url: str, repetitions: int) -> list:
    all_results = []
    test_id = 1

    # Group by scenario for nicer output
    current_scenario = None

    for scenario, agent, context, expected, state in TEST_CASES:
        if scenario != current_scenario:
            current_scenario = scenario
            print(f"\n{'='*70}")
            print(f"  SCENARIO: {scenario.upper()}  (agent: {agent})")
            print(f"{'='*70}")

        short_ctx = context[:55] + ("..." if len(context) > 55 else "")
        print(f"\n  [{test_id}] {short_ctx}")
        print(f"       state={state or 'default'}  expect={expected}")

        for rep in range(1, repetitions + 1):
            # 1. Clear any previous result for this id
            try:
                dali2_inject(url, agent, f"reset_results({test_id})")
            except Exception as e:
                print(f"    WARNING reset: {e}")

            # 2. Set agent state if needed (give the cycle time to process it)
            if state:
                try:
                    dali2_inject(url, agent, f"set_status({state})")
                    time.sleep(0.3)
                except Exception as e:
                    print(f"    WARNING set_status: {e}")

            # 3. Inject the test event and start timer
            t0 = time.time()
            try:
                dali2_inject(url, agent, f"run_test({test_id},{context})")
            except Exception as e:
                all_results.append(Result(
                    scenario=scenario, test_id=test_id, repetition=rep,
                    agent=agent, query=context, state=state,
                    expected=expected, outcome="error",
                    correct=False, latency_ms=0.0, error=str(e)
                ))
                test_id += 1
                continue

            # 4. Poll: blocked queries resolve very fast; LLM calls take seconds
            timeout = BLOCKED_TIMEOUT if expected == "blocked" else POLL_TIMEOUT
            outcome, latency_ms = poll_for_result(url, agent, test_id, timeout)

            if outcome is None:
                outcome = "timeout"

            correct = (
                outcome == expected
                or (expected == "accepted_or_rejected"
                    and outcome in ("accepted", "rejected"))
            )

            sym = "OK" if correct else "FAIL"
            print(f"    rep {rep}: [{sym}] {outcome:8s}  {latency_ms:.0f} ms")

            all_results.append(Result(
                scenario=scenario, test_id=test_id, repetition=rep,
                agent=agent, query=context, state=state,
                expected=expected, outcome=outcome,
                correct=correct, latency_ms=round(latency_ms, 1)
            ))
            test_id += 1
            time.sleep(0.2)

    return all_results


# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(results: list) -> dict:
    stats = {}
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r.scenario].append(r)

    for scenario, rlist in by_scenario.items():
        total    = len(rlist)
        blocked  = sum(1 for r in rlist if r.outcome == "blocked")
        accepted = sum(1 for r in rlist if r.outcome == "accepted")
        rejected = sum(1 for r in rlist if r.outcome == "rejected")
        errors   = sum(1 for r in rlist if r.outcome in ("error", "timeout"))
        correct  = sum(1 for r in rlist if r.correct)

        llm_list  = [r for r in rlist if r.outcome in ("accepted", "rejected")]
        lats      = [r.latency_ms for r in llm_list if r.latency_ms > 0]
        avg_lat   = sum(lats) / len(lats) if lats else 0
        min_lat   = min(lats) if lats else 0
        max_lat   = max(lats) if lats else 0

        filt_list = [r for r in rlist if r.outcome == "blocked"]
        filt_lats = [r.latency_ms for r in filt_list]
        avg_filt  = sum(filt_lats) / len(filt_lats) if filt_lats else 0

        stats[scenario] = {
            "total_tests":            total,
            "blocked":                blocked,
            "accepted":               accepted,
            "rejected":               rejected,
            "errors":                 errors,
            "filter_accuracy":        round(correct / total * 100, 1) if total else 0,
            "llm_calls":              len(llm_list),
            "avg_latency_ms":         round(avg_lat, 1),
            "min_latency_ms":         round(min_lat, 1),
            "max_latency_ms":         round(max_lat, 1),
            "avg_filter_overhead_ms": round(avg_filt, 3),
        }
    return stats


def print_summary(stats: dict):
    print(f"\n\n{'='*70}")
    print("  EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'='*70}\n")
    for scenario, s in stats.items():
        print(f"  Scenario: {scenario.upper()}")
        print(f"  {'-'*50}")
        print(f"  Total tests:         {s['total_tests']}")
        print(f"  Blocked (tell):      {s['blocked']}")
        print(f"  Accepted:            {s['accepted']}")
        print(f"  Rejected (told):     {s['rejected']}")
        print(f"  Errors/timeouts:     {s['errors']}")
        print(f"  Filter accuracy:     {s['filter_accuracy']:.1f}%")
        print(f"  LLM calls made:      {s['llm_calls']}")
        print(f"  Avg LLM latency:     {s['avg_latency_ms']} ms")
        print(f"  Min/Max latency:     {s['min_latency_ms']}/{s['max_latency_ms']} ms")
        print(f"  Avg filter overhead: {s['avg_filter_overhead_ms']} ms")
        print()


def save_results(results: list, stats: dict, url: str, model: str):
    output = {
        "metadata": {
            "dali2_url":   url,
            "model":       model,
            "repetitions": NUM_REPETITIONS,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "statistics": stats,
        "raw_results": [r.__dict__ for r in results],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DALI2 tell/told filter experiment â€” requires DALI2 running via Docker"
    )
    parser.add_argument("--url",  default=DEFAULT_DALI2_URL,
                        help=f"DALI2 server URL (default: {DEFAULT_DALI2_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--repetitions", type=int, default=NUM_REPETITIONS,
                        help=f"Repetitions per test case (default: {NUM_REPETITIONS})")
    args = parser.parse_args()
    NUM_REPETITIONS = args.repetitions

    print("=" * 70)
    print("  DALI2 Tell/Told Filter Experimental Evaluation")
    print(f"  DALI2:  {args.url}")
    print(f"  Model:  {args.model}")
    print(f"  Reps:   {NUM_REPETITIONS}")
    print("=" * 70)

    # Check DALI2 is reachable
    if not dali2_alive(args.url):
        print(f"\nERROR: Cannot reach DALI2 at {args.url}")
        print("Start DALI2 first:")
        print("  cd DALI2")
        print("  # Windows PowerShell:")
        print("  $env:OPENROUTER_API_KEY='sk-or-...'")
        print("  $env:AGENT_FILE='examples/experiment_agents.pl'")
        print("  docker compose up --build")
        sys.exit(1)

    # API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        try:
            api_key = getpass.getpass(
                "OpenRouter API key (or set OPENROUTER_API_KEY): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not provided.", file=sys.stderr)
        sys.exit(1)

    # Push key + model to DALI2 (propagated via Redis to all agent processes)
    dali2_set_ai_key(args.url, api_key)
    dali2_set_ai_model(args.url, args.model)
    print(f"\n  AI Oracle configured.  enabled={dali2_ai_enabled(args.url)}")

    results = run_experiment(args.url, NUM_REPETITIONS)
    stats   = compute_statistics(results)
    print_summary(stats)
    save_results(results, stats, args.url, args.model)

