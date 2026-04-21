# lgrl_final Pipeline Documentation

This document explains the lgrl_final pipeline file by file and method by method.

## 1. High-level architecture

lgrl_final is organized into these layers:

1. CLI and orchestration: `main.py`
2. Training and evaluation orchestration: `ppo_trainer.py`
3. Environment construction: `env_factory.py`
4. Gym wrappers: `wrappers.py`
5. Action masking strategy: `action_mask.py`
6. Subgoal strategies: `subgoal_manager.py`
7. Convergence metric callback: `callbacks.py`
8. Scenario metadata: `scenario_specs.py`
9. Observation to text translation: `observation_translator.py`

## 2. End-to-end pipeline

### 2.1 CLI execution flow

When you run:

`python -m lgrl_final.main ...`

this happens:

1. `main()` parses CLI args.
2. Seeds are parsed by `_parse_seed_list()`.
3. `ExperimentConfig` is created.
4. `ExperimentRunner(config)` is created.
5. `ExperimentRunner._configure_backends()` sets `utils.ENABLE_PENGYM` and `utils.ENABLE_NASIM`.
6. If PenGym backend is enabled, config and service-port map are initialized.
7. `PPOTrainer` is created.
8. `PPOTrainer.train()` builds env, model, and runs learning.
9. Optional model save via `PPOTrainer.save()`.
10. `PPOTrainer.evaluate()` runs evaluation episodes.
11. Summary is printed as text or JSON.

### 2.2 Dynamic vs static mode

1. Dynamic mode (`--scenario-name`): reseedable env factory path.
2. Static mode (`--scenario-path`): static env factory path.
3. Both together are rejected.

### 2.3 Masked vs unmasked ablation mode

1. Default: masking enabled.
2. With `--disable-action-masking`: masking disabled in both training and evaluation.

## 3. File-by-file and method-by-method

## main.py

Purpose: CLI entrypoint and experiment orchestration.

### `ExperimentConfig`

Dataclass that carries all run settings:

1. Agent and scenario selectors.
2. Timesteps and episode controls.
3. Backend toggles (PenGym/NASIM).
4. Action masking toggle.
5. Subgoal manager mode.

### `ExperimentRunner.__init__(config)`

1. Stores config.
2. Calls `_configure_backends()`.
3. Creates `PPOTrainer` with manager mode, masking mode, optional LLM dependencies.

### `ExperimentRunner._configure_backends()`

1. Computes scenario mode.
2. Picks default backend behavior:

- Dynamic mode defaults to NASIM simulation.
- Static mode defaults to PenGym backend.

3. Applies explicit overrides (`enable_pengym`, `enable_nasim`) if provided.
4. Validates at least one backend remains enabled.
5. Writes backend flags to `pengym.utilities` globals.
6. If PenGym is enabled, initializes config and service-port map.

### `ExperimentRunner.run()`

1. Calls `trainer.train(...)`.
2. Saves model unless disabled or loading existing model.
3. Calls `trainer.evaluate(...)`.
4. Returns a result dictionary.

### `_parse_seed_list(seed_text)`

1. Converts comma-separated seed string to list of ints.
2. Returns `None` if empty.
3. Raises clear error if any token is not an integer.

### `_build_parser()`

Builds argparse CLI with:

1. Agent type.
2. Mutually exclusive scenario selectors.
3. Config/model/timestep/eval controls.
4. Subgoal manager mode.
5. Action masking toggle.
6. Backend toggles.
7. Save/output format options.

### `main(argv=None)`

1. Parses args.
2. Parses seed lists.
3. Warns if LLM manager selected without injected client from API.
4. Builds `ExperimentConfig`.
5. Runs `ExperimentRunner`.
6. Prints summary.
7. Returns process exit code.

## ppo_trainer.py

Purpose: training/evaluation orchestration around MaskablePPO.

### `PPOTrainer.__init__(...)`

1. Validates agent type.
2. Stores hyperparameters and runtime options.
3. Validates subgoal manager mode.
4. Builds subgoal manager if agent type is LGRL:

- Deterministic manager.
- LLM manager (with deterministic fallback behavior).

5. Creates `EnvFactory` with masking mode.

### `_resolve_scenario_inputs(scenario_name, scenario_path, allow_fallback=False)`

1. Optionally falls back to stored scenario values.
2. Enforces exactly one scenario selector.
3. Returns tuple:

- `scenario_key` (name or stem of path)
- resolved name
- resolved path

### `_build_model(vec_env)`

Creates and returns a `MaskablePPO` model with configured hyperparameters.

### `load(model_path)`

Loads model from disk into `self.model`.

### `train(...)`

1. Resolves scenario mode.
2. Stores scenario state in trainer.
3. If `model_path` provided: loads model and returns.
4. Gets env factory function from `EnvFactory`.
5. Wraps in `DummyVecEnv` and `VecFrameStack`.
6. Builds model.
7. Creates `ConvergenceCallback` if ideal steps exist.
8. Calls `model.learn(...)` with `use_masking` according to ablation toggle.
9. Stores convergence speed if available.
10. Returns model.

### `evaluate(...)`

1. Ensures model exists.
2. Resolves scenario mode.
3. Picks seeds and episodes.
4. For each seed:

- Builds eval env.
- Runs episode loop.
- Chooses masked or unmasked `predict()` path.
- Steps env and accumulates stats.

5. Prints final summary metrics.
6. Returns final `(done, truncated, ep_steps)` from last episode.

### `save(output_dir="models")`

1. Ensures model exists.
2. Resolves scenario key for file naming.
3. Creates timestamped model name.
4. Saves model and returns path string.

## env_factory.py

Purpose: build gym environments for training and evaluation.

### `EnvFactory.__init__(...)`

Stores:

1. Agent mode.
2. Subgoal manager.
3. Episode length.
4. Masking mode and function.
5. Intrinsic reward options.

### `create_pengym_env(scenario_name, seed=None)`

Creates generated scenario env via `pengym.create_environment`.

### `create_pengym_custom_environment(scenario_path)`

Creates static/custom scenario env via `pengym.load`, and seeds numpy/action space.

### `_build_env_kwargs()`

Builds wrapper options (LGRL guidance and intrinsic reward options).

### `_apply_wrappers(env, ...)`

Applies wrappers in this order:

1. `IntActionWrapper`
2. `TimeLimit`
3. LGRL wrappers (optional):

- `SubgoalUpdateWrapper`
- `SubgoalObsWrapper`
- `SubgoalRewardWrapper` (optional)

4. `Monitor`
5. `ActionMasker` (optional; only if masking enabled)

### `normalize_firewall_collections(env)`

Converts firewall values to lists for compatibility with generated scenarios.

### `build_env(scenario_name=None, scenario_path=None, seed=None)`

1. Enforces exactly one selector.
2. Creates base env from name or path.
3. Applies wrappers.
4. Normalizes firewall collections.
5. Returns final wrapped env.

### `make_eval_env(scenario_name, seed)`

Returns closure that builds dynamic eval env for a fixed seed.

### `make_static_env(scenario_path)`

Returns closure that builds static env from path.

### `make_env_reseedable(scenario_name, seed_pool)`

Returns nested `_ReseedableEnv` class:

1. Samples seed at init/reset.
2. Rebuilds env each reset.
3. Delegates step/render.
4. Exposes `action_masks()` for vec env compatibility.
5. If masking disabled, returns all-true mask.
6. Delegates unknown attributes via `get_wrapper_attr`.

### `build_train_env_factory(...)`

1. Enforces selector validity.
2. Static path => static factory.
3. Dynamic name => reseedable factory.

## wrappers.py

Purpose: reusable wrappers for action normalization and LGRL signals.

### `one_hot_subgoal(subgoal)`

Converts subgoal string to fixed one-hot vector.

### `IntActionWrapper.action(action)`

Normalizes single-element numpy actions into python ints.

### `SubgoalObsWrapper.__init__(env, subgoal_manager)`

Expands observation space by appending one-hot subgoal vector.

### `SubgoalObsWrapper.observation(obs)`

Returns concatenated observation + current subgoal vector.

### `SubgoalRewardWrapper.__init__(env, subgoal_manager, lambda_=0.5)`

Stores manager and intrinsic reward weight.

### `SubgoalRewardWrapper.reward(reward)`

Adds intrinsic reward if manager reports recent completion.

### `SubgoalUpdateWrapper.__init__(env, subgoal_manager)`

Stores manager reference.

### `SubgoalUpdateWrapper.reset(**kwargs)`

Resets subgoal manager and passes reset through.

### `SubgoalUpdateWrapper.step(action)`

Steps env, then updates manager state machine.

## action_mask.py

Purpose: implement action validity rules for masked PPO.

### `CustomActionMask.mask_fn(env)`

1. Gets base env and current state.
2. Iterates over action space.
3. Invalidates actions targeting unknown hosts.
4. Allows subnet scans and privilege escalation only with USER+ access.
5. For other actions, requires discovered target host.
6. If all false, forces all true to avoid invalid distribution.

## subgoal_manager.py

Purpose: strategy objects for subgoal progression.

### Constants

1. `SUBGOALS`: ordered allowed goals.
2. `SUBGOAL_TO_IDX`: one-hot mapping.

### `BaseSubgoalManager`

Abstract interface with:

1. `reset()`
2. `update()`
3. `get()`

### `DeterministicSubgoalManager.__init__(utils, storyboard)`

Stores PenGym/NASIM state handles and initializes counters.

### `DeterministicSubgoalManager.reset()`

Initializes subgoal and previous host/shell counters.

### `DeterministicSubgoalManager._extract_counts()`

Extracts counts from either:

1. live NASIM state (`current_state`), or
2. PenGym host maps.

### `DeterministicSubgoalManager.update()`

1. Updates transition flags.
2. Applies deterministic state machine for next subgoal selection.
3. Stores updated counters.

### `LLMSubgoalManager.__init__(...)`

Stores llm client and translator, and creates deterministic fallback manager.

### `LLMSubgoalManager.reset()`

Resets fallback and syncs current subgoal from fallback.

### `LLMSubgoalManager._parse_subgoal(text)`

Extracts first known subgoal token from model output.

### `LLMSubgoalManager._build_prompt()`

Builds constrained prompt with allowed subgoals and optional translated context.

### `LLMSubgoalManager._query_llm(prompt)`

Supports multiple client interfaces:

1. callable
2. `.generate`
3. `.complete`
4. `.invoke`

Returns normalized text response.

### `LLMSubgoalManager.update()`

1. Always updates fallback first.
2. If no transition occurred, keep fallback subgoal.
3. If transition occurred, ask LLM and parse response.
4. If parsing fails, fallback subgoal is used.

## callbacks.py

Purpose: convergence speed metric during training.

### `ConvergenceCallback.__init__(ideal_steps, window_size=100, margin=2, verbose=0)`

Stores threshold parameters and runtime counters.

### `ConvergenceCallback._on_step()`

1. Increments episode length counter.
2. On episode end (non-truncated), appends length.
3. Computes rolling mean of recent episodes.
4. Marks first timestep where rolling mean <= `ideal_steps + margin`.
5. Resets episode counter for next episode.

## scenario_specs.py

Purpose: static metadata for generated scenarios used by translator.

### `SCENARIO_SPECS`

Maps scenario name to dimensions and counts.

### `get_scenario_spec(scenario_name)`

Returns spec dictionary or raises clear error listing available scenarios.

## observation_translator.py

Purpose: convert flat numeric observation vectors into readable text.

### `ObservationTranslator.__init__(observation=None, scenario=None)`

Stores optional default observation and scenario.

### `update(observation)`

Replaces stored observation.

### `get_detail(scenario=None)`

Returns scenario metadata using `get_scenario_spec`.

### `translate(observation=None, scenario=None)`

1. Resolves observation and scenario.
2. Validates inputs.
3. Converts to float numpy vector.
4. Validates 1D shape.
5. Reshapes into host rows.
6. Formats summary text.

### `get_description()`

Caches and returns current translation.

### `_host_row_size(cfg)`

Computes per-host row width from scenario spec.

### `_reshape_flat_observation(flat_obs, cfg)`

Validates total length and reshapes to `(hosts, row_size)`.

### `_format_text(host_rows, cfg, scenario_name)`

Builds multi-line report including host details and aggregate counts.

### `_decode_host_row(row, cfg)`

Decodes one host row into semantic fields (flags, values, one-hot slices).

### `_decode_single_onehot(vec, prefix)`

Turns one-hot slice into symbolic token.

### `_decode_multi_hot(vec, prefix)`

Turns multi-hot slice into comma-separated symbolic tokens.

### `_access_to_label(access)`

Maps numeric access to NONE/USER/ROOT label.

## 4. Practical run examples

### 4.1 Dynamic mode, masking on

`python -m lgrl_final.main --agent-type ppo --scenario-name tiny-gen`

### 4.2 Dynamic mode, masking off (ablation)

`python -m lgrl_final.main --agent-type ppo --scenario-name tiny-gen --disable-action-masking`

### 4.3 Static mode from scenario file

`python -m lgrl_final.main --agent-type ppo --scenario-path database/scenarios/medium-multi-site.yml`

### 4.4 LGRL deterministic manager

`python -m lgrl_final.main --agent-type lgrl --scenario-name tiny-gen --subgoal-manager-type deterministic`

### 4.5 LGRL LLM manager (CLI fallback note)

`python -m lgrl_final.main --agent-type lgrl --scenario-name tiny-gen --subgoal-manager-type llm`

Note: CLI currently does not inject a concrete LLM client object, so LLM manager behavior falls back unless using the Python API with `ExperimentConfig.llm_client`.

## 5. Current caveats

1. For generated dynamic scenarios, convergence warning can appear if scenario key is absent from `IDEAL_STEPS`.
2. `MaskablePPO` still requires compatible installed sb3/sb3-contrib versions in your runtime environment.
3. Evaluation returns the final episode status tuple plus printed aggregate metrics.
