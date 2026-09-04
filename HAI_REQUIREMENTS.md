# Requirements for a Diagnostically Extended HAI-CPPS Version

## Purpose and Scope

This document describes changes to the simulation, metadata, and export format
of HAI-CPPS that are needed for causally interpretable diagnosis with virtual
observability. It does **not** describe any post-hoc manipulation of measurement
series. The measurement channels published so far remain unchanged. In
addition, simulator-internal states that are not observable online should be
exported as separate oracle data, together with complete structural knowledge.

The separation must be strict, both technically and conceptually:

- `measurements`: sensors and control variables that are actually available in
  diagnostic operation;
- `oracle_states`: simulator-internal states that may be used as targets during
  training on normal runs and for offline evaluation, but never as online input;
- `fault_metadata`: fault location, fault type, injection time, and causal
  events; labels/metadata only;
- `system_knowledge`: modules, ports, directed edges, state equations,
  qualitative rules, and timing bounds;
- `technical_timing`: sampling, communication, sensor, and actuator delays with
  uncertainty bounds.

Without this separation, it is impossible to verify whether a neural model
reconstructs a physically interpretable state or merely uses a freely rotatable
hidden space.

## Why the Current Version Makes Diagnosis Cases Difficult

The previous experiments show several independent identifiability problems:

1. **Missing causal intermediate states.** The 100 published process channels
   used in the experiments show pressures, fill levels, volume flows, and
   discrete phases, but not consistently the internal states through which a
   local fault propagates to these measured variables. Examples include
   hydraulic resistance, effective valve opening, pump delivery state, filter
   loading, material in transport, and thermal holdup.
2. **Shared operating phases dominate local fault signatures.** In `ds10`,
   earlier runs repeatedly showed the same early bottling phase activity for
   very different causes. As a result, a synchronous or phase-induced downstream
   state was incorrectly treated as local root-cause evidence.
3. **Topological propagation without complete runtime information.** A
   downstream module can record a symptom earlier than the true source module if
   sensor, process, and logging delays are mixed together. A small global time
   correction does not solve this problem.
4. **Local cause without a directly visible channel.** If, for example, a clog
   primarily changes a non-exported pipe resistance, the labeled fault location
   may not be uniquely identifiable from the published variables. Good global
   anomaly detection then does not guarantee a good module ranking.
5. **Threshold-induced candidate deletion.** Earlier evaluations only included
   modules in the ranking if they crossed a local persistent threshold. The true
   module could therefore be missing even though a downstream or coupled module
   was clearly anomalous. This is an evaluation problem, but complete causal
   states would also improve the direct evidence.
6. **`bottling0` clogging is currently a separate identifiability question.**
   Seven cases were excluded in advance in V6: validation `ds5`, `ds6`, `ds7`,
   `ds9`; test `ds4`, `ds8`, `ds10`, each with
   `bottling0_anom_clogging`. Other bottling faults such as `pump50`, `pump75`,
   or valve faults must not be removed along with them. A new dataset version
   must show which internal state is changed by "clogging", where this change is
   physically located, and which locally or downstream observable consequences
   follow from it.

## Additional States That Must Be Exported

The final names should be taken from the actual Modelica/simulation components.
The following stable semantic roles are required regardless of the concrete
naming convention.

### For Each Process Module

| State | Meaning | Required structural inputs | Recommended unit |
|---|---|---|---|
| `hidden.inventory` | Material/liquid content of the module | Port mass flows, tank states, initial content | kg or m3 |
| `hidden.net_inventory_flow` | Sum of inflows that are positive under the port convention | all `port_*.m_flow` values with documented sign | kg/s |
| `hidden.effective_throughput` | Quantity actually transported per unit time | Pump, valve, and port states | kg/s or m3/s |
| `hidden.operation_phase` | Unique internal process state | State machine/mode ID | categorical |
| `hidden.actuator_command` | Requested actuator value | Controller/PLC output | normalized or SI |
| `hidden.actuator_effective` | Physically effective actuator value | Command, dynamics, saturation, fault state | normalized or SI |
| `hidden.local_resistance` | Effective hydraulic or transport-related resistance | Geometry, valve, clogging, filter state | documented SI quantity |
| `hidden.local_capacity` | Maximum possible local delivery/processing rate | Pump curve, valve, fill level, resistance | kg/s or m3/s |

### Mixer

- actual content and composition of each mixing tank, if simulated;
- net inflow and outflow of each tank;
- effective position of each inlet and outlet valve;
- internal pump state, including requested and achieved delivery rate;
- active recipe/mixing phase and its transition condition;
- internal pipe or outlet clogging as a resistance parameter, not only as a
  fault label.

### Distillation

- liquid and vapor holdup;
- thermal energy state, heater power, and effective heat transfer;
- boiling, condensation, or separation rate;
- composition/quality state, if present in the model;
- effective inflows and outflows, as well as internal hydraulic resistances;
- internal pump and valve state;
- active distillation phase with local phase start.

### Filter

- accumulated filter loading or retained mass;
- permeability and effective filter resistance;
- pressure drop across the filter;
- actual flow capacity;
- contamination/clogging state as a continuous physical state;
- effective pump and valve states.

### Bottling

- buffer content upstream of the filler;
- pipe/nozzle content and effective pipe resistance;
- internal nozzle or clogging opening;
- requested and effective pump delivery;
- requested and effective valve opening;
- local pressure difference and flow capacity;
- bottle presence, filling phase, and actual fill quantity;
- unambiguous position of the `clogging` fault: inlet, tank outlet, pipe, or
  nozzle.

### For Each Directed Connection

| State/metadata | Purpose |
|---|---|
| `edge.<source>.<target>.material_in_transit` | non-observable transport state between modules |
| `edge.<source>.<target>.source_flow` | physical output at the source port |
| `edge.<source>.<target>.target_flow` | physical input at the target port |
| `edge.<source>.<target>.transport_delay` | runtime-dependent real process delay |
| `edge.<source>.<target>.pressure_drop` | local coupling/resistance information, where meaningful |
| `edge.<source>.<target>.compatibility_residual` | balance or coupling residual under correct time alignment |

## Formal System Knowledge File

For each `ds` setup, a separate machine-readable knowledge file must be
generated. Edges or modules must not be merged across different setups. Every
entry for a non-observable variable requires at least:

```yaml
- name: hidden.filter0.filter_loading
  module: filter0
  kind: filter_loading
  online_observable: false
  oracle_column: filter0.oracle.filter_loading
  inputs:
    - filter0.port_in.m_flow
    - filter0.port_out.m_flow
    - filter0.oracle.filter_loading_previous
  function: integrate_weighted_sum
  coefficients: [1.0, -1.0, 1.0]
  unit: kg
  valid_modes: [filtering]
  lower_bound: 0.0
  provenance: Modelica component and equation identifier
```

The function library must be versioned and must support at least weighted
sums/balances, differences, time-delayed differences, discrete state
transitions, products, saturations, and integrators. For every equation, the
unit, sign convention, valid mode, and parameter source must be specified. A
purely statistical correlation is **not** a structural function.

## Document Normal Knowledge and Fault Knowledge Separately

For every rule, the following must be specified:

- whether it remains invariant during a fault (`K_imp`-compatible);
- whether it only describes normal behavior (`K_score` or interpretation);
- whether it is used to reconstruct a specific target;
- which variables it affects as conclusions;
- in which operating modes and value ranges it is valid;
- which delay and uncertainty are admissible.

Balance equations, port signs, and deterministic state equations are typically
`K_imp`-compatible. "Signal is nominal", normal correlations, and normal phase
sequences must not be used for imputation in the fault case, because they may
regularize away the anomaly.

## Timing and Causality Metadata

Every run requires at least the following timestamps on a shared simulation time
axis:

- intended fault injection time;
- time of the actual physical parameter change;
- first time of each changed internal state variable;
- first time of each measurable local consequence;
- first time of each consequence at downstream modules;
- sensor sampling, communication, and logging delay;
- upper and lower delay bound per edge and channel.

This makes it possible to evaluate separately whether (i) the cause acts
physically first, (ii) a sensor detects it first, and (iii) a logger records it
first. Only under validated bounds can an unambiguous causal cause be inferred
from the earliest recorded anomaly.

## Fault Design and Identifiability

Three controls must be provided for every fault class:

1. **local signature:** at least one local internal state must change after
   injection;
2. **propagation:** the expected causal chain and its delay intervals must be
   traceable in the oracle export;
3. **distinguishability:** faults from different modules must not produce only
   the same global phase change without additional local information.

In addition, paired or contrast runs with identical operating conditions and
only one varied fault parameter must be provided. For `bottling0` clogging, a
separate include/exclude sensitivity study is required. If the class cannot be
distinguished from the normal case or from other bottling faults even with
complete oracle states, it must be marked as non-identifiable instead of forcing
an unambiguous module diagnosis.

## Recommended File Layout

```text
<scenario>/
  measurements.parquet
  oracle_states.parquet
  fault_events.json
  system_knowledge.yaml
  technical_timing.json
  sim_setup.json
  provenance.json
```

All tables share `scenario_id`, `simulation_step`, and `simulation_time`.
`oracle_states` must not be automatically merged with `measurements` when
exporting the deployable input matrix. Fault flags, fault severity, component
name, and fault time are leakage metadata and must never be used as model
features.

## Acceptance Criteria for a New HAI-CPPS Version

- Every published root-cause class changes a documented local internal state.
- Every non-observable variable has a name, module, unit, inputs, function, and
  provenance.
- All setups have separate complete directed graphs with port assignments.
- Normal, oracle, and fault data are technically separated.
- Timing bounds are validated empirically or from the simulation.
- `bottling0` clogging is either made identifiable or explicitly documented as
  non-identifiable.
- An automated leakage test proves that oracle and fault columns do not enter
  online inputs.
- An automated equation test checks the exported states against their specified
  functions and units.
