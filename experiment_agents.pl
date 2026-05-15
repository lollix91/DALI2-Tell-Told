%% DALI2 Experiment Agents — Tell/Told Filtering with AI Oracle
%% ============================================================
%%
%% This file defines three agents used to evaluate DALI2's tell/told
%% communication filtering mechanism applied to LLM (AI Oracle) interactions.
%% It mirrors the three scenarios from the Python simulation in experiment.py.
%%
%% Agents:
%%   crop_advisor   — Smart Agriculture (Section 4.1 of the paper)
%%   coordinator    — Emergency Response (Section 4.2 of the paper)
%%   state_test     — State-Dependent Filtering (Experiment Section)
%%
%% Running with Docker:
%%   set OPENROUTER_API_KEY=sk-or-...   (Windows)
%%   export OPENROUTER_API_KEY=sk-or-... (Linux/macOS)
%%
%%   docker compose -f ../DALI2/docker-compose.yml up --build ^
%%     --env AGENT_FILE=../DALI2-Tell-Told/experiment_agents.pl
%%
%% Or copy this file to the DALI2 examples/ directory and run:
%%   AGENT_FILE=examples/experiment_agents.pl docker compose up --build
%%
%% Then send test events via the Web UI at http://localhost:8080 or REST API.
%%
%% Test event syntax (send to the agent using the /api/send endpoint):
%%   {"to":"crop_advisor","content":"test_filter(soil_analysis(moisture(25),ph(6_5),field(north)))"}
%%   {"to":"crop_advisor","content":"set_status(idle)"}
%%
%% Results are stored as beliefs: test_result(Context, Outcome)
%% Read them at: GET http://localhost:8080/api/beliefs?agent=crop_advisor


%% ============================================================
%% SCENARIO A: SMART AGRICULTURE — crop_advisor
%% ============================================================
%%
%% Tell rules: only soil_analysis/3 and weather_analysis/3 may be sent to AI.
%% Told rules: only suggestion/1 (pri 80) and recommendation/2 (pri 90,
%%             only when status is active) are accepted from AI.
%%
%% Expected behaviour:
%%   test_filter(soil_analysis(...))   → ask_ai called; response accepted if
%%                                       matches suggestion/1 or recommendation/2
%%   test_filter(market_price(wheat))  → blocked by tell filter (no LLM call)

:- agent(crop_advisor, [cycle(1)]).

%% Initial belief
believes(status(active)).

%% ---- Tell filter ----
tell(_, _, soil_analysis(_,_,_)) :- true.
tell(_, _, weather_analysis(_,_,_)) :- true.

%% ---- Told filter ----
told(_, suggestion(_), 80) :- true.
told(_, recommendation(_,_), 90) :- believes(status(active)).

%% ---- Test handlers ----
%% run_test(Id, Context): exercises the filter+oracle pipeline.
%%   Stores test_result(Id, Outcome) where Outcome is: accepted | rejected | blocked.
%%   Python script injects this and polls /api/beliefs for the result.
run_testE(Id, Context) :>
    ask_ai(Context, Result),
    ( Result = blocked(_) ->
        Outcome = blocked
    ; Result = rejected(_) ->
        Outcome = rejected
    ;   Outcome = accepted
    ),
    assert_belief(test_result(Id, Outcome)),
    log("crop_advisor run_test ~w: ~w -> ~w", [Id, Context, Outcome]).

%% reset_results(Id): clears a previous test_result(Id,_) belief.
reset_resultsE(Id) :>
    ( retract_belief(test_result(Id, _)) -> true ; true ),
    log("crop_advisor: results reset for test ~w", [Id]).

%% ---- State control ----
%% Send set_status(active) or set_status(idle) to change agent beliefs.
set_statusE(Status) :>
    retract_belief(status(_)),
    assert_belief(status(Status)),
    log("crop_advisor: status set to ~w", [Status]).

%% ---- Production rules (from agriculture.pl) ----
soil_reportE(Moisture, PH, Field) :>
    ( ai_available ->
        ask_ai(
            soil_analysis(moisture(Moisture), ph(PH), field(Field)),
            Advice),
        log("AI crop advice: ~w", [Advice])
    ; true ),
    ( Moisture < 30 ->
        send(irrigation_controller, irrigate(Field))
    ; Moisture > 80 ->
        send(irrigation_controller, reduce_water(Field))
    ; log("Soil normal for ~w", [Field]) ).

weather_alertE(Temp, _Humidity, _Forecast) :>
    ( ai_available ->
        ask_ai(
            weather_analysis(temp(Temp), humidity(_), forecast(_)),
            Advice),
        log("AI weather advice: ~w", [Advice])
    ; true ),
    ( Temp > 38 ->
        send(irrigation_controller, irrigate(all_fields))
    ; Temp < 2 ->
        send(farmer_agent, advisory(frost_warning, all_fields))
    ; true ).


%% ============================================================
%% SCENARIO B: EMERGENCY RESPONSE — coordinator
%% ============================================================
%%
%% Tell rules: only analyze/1 and log_event/3 may be sent to AI.
%% Told rules: emergency/2 (pri 200), alert/2 (pri 100),
%%             sensor_data/1 (pri 30, active only), calibration_request (pri 10).
%%
%% Expected behaviour:
%%   test_filter(analyze(emergency(fire,building_a))) → ask_ai called;
%%     response likely rejected (LLM returns suggestion/1 which is not in whitelist)
%%   test_filter(send_email(admin,status))            → blocked by tell filter

:- agent(coordinator, [cycle(2)]).

%% Initial belief
believes(status(active)).

%% ---- Tell filter ----
tell(_, _, analyze(_)) :- true.
tell(_, _, log_event(_,_,_)) :- true.

%% ---- Told filter (priority queue) ----
told(_, emergency(_,_), 200) :- true.
told(_, alert(_,_), 100) :- true.
told(_, sensor_data(_), 30) :- believes(status(active)).
told(_, calibration_request, 10) :- true.

%% ---- Test handlers ----
run_testE(Id, Context) :>
    ask_ai(Context, Result),
    ( Result = blocked(_) ->
        Outcome = blocked
    ; Result = rejected(_) ->
        Outcome = rejected
    ;   Outcome = accepted
    ),
    assert_belief(test_result(Id, Outcome)),
    log("coordinator run_test ~w: ~w -> ~w", [Id, Context, Outcome]).

reset_resultsE(Id) :>
    ( retract_belief(test_result(Id, _)) -> true ; true ),
    log("coordinator: results reset for test ~w", [Id]).

%% ---- State control ----
set_statusE(Status) :>
    retract_belief(status(_)),
    assert_belief(status(Status)),
    log("coordinator: status set to ~w", [Status]).

%% ---- Production rules ----
emergencyE(Type, Value) :>
    log("EMERGENCY: ~w = ~w", [Type, Value]),
    ( ai_available ->
        ask_ai(analyze(emergency(Type, Value)), Advice),
        log("AI emergency advice: ~w", [Advice])
    ; true ).

sensor_dataE(Reading) :>
    log("Sensor data: ~w", [Reading]).


%% ============================================================
%% SCENARIO C: STATE-DEPENDENT FILTERING — state_test
%% ============================================================
%%
%% Tell rules: only suggestion_request/1 may be sent to AI.
%% Told rules: suggestion/1 and recommendation/2 accepted ONLY when active.
%%
%% Expected behaviour (state=active):
%%   test_filter(suggestion_request(...)) → ask_ai called → accepted
%% Expected behaviour (state=idle):
%%   test_filter(suggestion_request(...)) → ask_ai called → REJECTED by told
%% Off-domain:
%%   test_filter(check_stock_market(...)) → blocked by tell filter

:- agent(state_test, [cycle(1)]).

%% Initial belief: idle (to start with rejections; set to active to accept)
believes(status(idle)).

%% ---- Tell filter ----
tell(_, _, suggestion_request(_)) :- true.

%% ---- Told filter (state-dependent) ----
told(_, suggestion(_), 50) :- believes(status(active)).
told(_, recommendation(_,_), 90) :- believes(status(active)).

%% ---- Test handlers ----
run_testE(Id, Context) :>
    ask_ai(Context, Result),
    ( Result = blocked(_) ->
        Outcome = blocked
    ; Result = rejected(_) ->
        Outcome = rejected
    ;   Outcome = accepted
    ),
    assert_belief(test_result(Id, Outcome)),
    log("state_test run_test ~w: ~w -> ~w", [Id, Context, Outcome]).

reset_resultsE(Id) :>
    ( retract_belief(test_result(Id, _)) -> true ; true ),
    log("state_test: results reset for test ~w", [Id]).

%% ---- State control ----
set_statusE(Status) :>
    retract_belief(status(_)),
    assert_belief(status(Status)),
    log("state_test: status set to ~w", [Status]).
