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

# lgrl_final Technical Documentation

This README documents the full lgrl_final code path (training, evaluation, wrappers, subgoal logic, and translation utilities) based on the current implementation.

## Scope

This folder contains an object-oriented RL pipeline on top of PenGym/NASIM.

- Entry point and run orchestration: main.py
- Training and evaluation orchestration: ppo_trainer.py
- Environment construction and wrapper stacking: env_factory.py
- Action masking policy: action_mask.py
- Subgoal orchestration for LGRL: subgoal_manager.py
- Gym wrappers for subgoal observation/reward/update: wrappers.py
- Convergence callback: callbacks.py
- Observation-to-text translator: observation_translator.py
- Scenario metadata used by translator: scenario_specs.py

## Architecture Overview

The runtime path is:

1. Parse CLI args and build ExperimentConfig.
2. Select backend defaults (PenGym or NASIM simulation) in ExperimentRunner.
3. Build PPOTrainer.
4. Build train environment factory from EnvFactory.
5. Train MaskablePPO.
6. Optionally save model.
7. Build eval environments and run evaluation episodes.
8. Emit summary metrics.

Core design details:

- Two scenario modes are mutually exclusive:
  - Dynamic/reseeded: scenario_name.
  - Static/custom file: scenario_path.
- Action masking is enabled by default and can be disabled for ablations.
- LGRL behavior is fixed by trainer contract:
  - Training uses deterministic subgoal manager.
  - Evaluation uses LLM subgoal manager with deterministic fallback.

## CLI and Orchestration (main.py)

### ExperimentConfig

ExperimentConfig stores all runtime options including:

- Agent type (ppo or lgrl).
- Scenario selector.
- Seed lists for train/eval.
- Training/evaluation limits.
- Backend toggles (PenGym/NASIM).
- Subgoal manager mode and action masking mode.
- Optional llm_client and translator injection for Python API usage.

### ExperimentRunner.\_configure_backends

Backend defaults are mode-aware:

- Dynamic mode (scenario_name) defaults to NASIM simulation on.
- Static mode (scenario_path) defaults to PenGym on.

Then explicit overrides are applied via enable_pengym and enable_nasim.

Validation rule:

- At least one backend must remain enabled.

When PenGym is enabled, it initializes CONFIG.yml and service-port mapping using pengym.utilities.

### ExperimentRunner.run

Run sequence:

1. trainer.train(...)
2. Optional trainer.save(...)
3. trainer.evaluate(...)
4. Return model handle, model path, raw episode status, and metrics.

### CLI arguments summary

- Scenario selection (required, mutually exclusive):
  - --scenario-name
  - --scenario-path
- Agent type:
  - --agent-type {ppo,lgrl}
- Optional training/eval controls:
  - --total-timesteps
  - --eval-episodes
  - --train-seeds
  - --eval-seeds
- Optional model behavior:
  - --model-path
  - --no-save
  - --model-output-dir
- Optional backend/masking behavior:
  - --disable-action-masking
  - --disable-pengym
  - --nasim-simulation
- Output:
  - --json

## Trainer Internals (ppo_trainer.py)

### Trainer responsibilities

PPOTrainer handles:

- Scenario input validation.
- Train/eval environment creation.
- MaskablePPO model construction/loading/saving.
- Convergence and training metrics.
- Evaluation metrics aggregation.

### Subgoal manager wiring

- Agent type ppo:
  - No subgoal manager in wrappers.
- Agent type lgrl:
  - train_subgoal_manager = DeterministicSubgoalManager
  - eval_subgoal_manager = LLMSubgoalManager
  - Separate EnvFactory instances are created for train and eval.

This split is intentional and enforced by code.

### Training flow

train(...):

1. Resolve scenario_name/scenario_path.
2. If model_path is provided, skip training and load model.
3. Build train env factory using EnvFactory.build_train_env_factory(...).
4. Wrap with DummyVecEnv and VecFrameStack.
5. Build MaskablePPO with configured hyperparameters.
6. Attach ConvergenceCallback using scenario-specific IDEAL_STEPS when available.
7. Learn with use_masking flag matching current mode.
8. Store convergence and return metrics.

### Evaluation flow

evaluate(...):

1. Require an available model.
2. Resolve scenario inputs (allowing fallback to stored train scenario).
3. Determine seeds and episodes-per-seed.
4. For each seed and episode:
   - Build eval env.
   - Reset and step until terminal/truncated/max_steps.
   - Predict actions with or without action masks.
   - Aggregate episode reward, steps, success, and LLM token usage.
5. Compute aggregate metrics:
   - success_rate
   - average_steps
   - average_cumulative_reward
   - average_token_usage

Return value currently returns only the final episode tuple: (done, truncated, ep_steps). Full aggregates are stored in last_eval_metrics and printed.

## Environment Factory and Wrapper Stack (env_factory.py, wrappers.py)

### Environment creation modes

- create_pengym_env(scenario_name, seed): for dynamic generated scenarios.
- create_pengym_custom_environment(scenario_path): for static custom scenario files.

### Wrapper order (important)

Wrappers are applied in this order:

1. IntActionWrapper
2. TimeLimit
3. If LGRL mode:
   - SubgoalUpdateWrapper
   - SubgoalObsWrapper
   - SubgoalRewardWrapper (optional intrinsic reward)
4. Monitor
5. ActionMasker (if masking enabled)

ActionMasker is intentionally outermost to keep action_masks discoverable through wrapper traversal in vectorized environments.

### Reseedable training environment

For dynamic training, make_env_reseedable(...) returns a custom nested env class that:

- Samples a random seed from a pool on init and each reset.
- Rebuilds the full wrapped env each episode.
- Exposes action_masks() and delegates attributes through get_wrapper_attr(...), preserving wrapper-chain compatibility.

## Action Masking Rules (action_mask.py)

CustomActionMask.mask_fn(env) applies per-action validity checks:

1. Reject actions with targets absent in host_num_map.
2. subnet_scan actions require USER access on target host.
3. pe\_ actions require USER access on target host.
4. All other actions require discovered target host.
5. If all actions become invalid, fallback to all-true mask to avoid invalid sampling state.

## Subgoal Managers (subgoal_manager.py)

### DeterministicSubgoalManager

Implements a state-machine style subgoal progression based on discovered hosts and shell counts.

- Reads either NASIM-like current_state or PenGym-style host_map structures.
- Tracks previous counts across steps.
- Sets just_completed when a subgoal transition condition is met.

### LLMSubgoalManager

Wraps deterministic behavior and only queries the LLM at transition points:

1. Always update deterministic fallback first.
2. If no transition happened, keep deterministic subgoal.
3. If transition happened:
   - Build constrained prompt from allowed subgoals.
   - Optionally add translated observation context.
   - Query llm_client if provided.
   - Parse response to one valid subgoal token.
4. If parsing/client fails, deterministic fallback subgoal is used.

Supported llm_client interfaces:

- Callable: llm_client(prompt)
- Method: generate(prompt)
- Method: complete(prompt)
- Method: invoke(prompt)

Token accounting:

- Attempts to read usage.total_tokens, or prompt_tokens + completion_tokens.
- Accumulates per-episode token usage via get_episode_token_usage().

## Observation Translation (observation_translator.py, scenario_specs.py)

ObservationTranslator converts a flat 1D observation vector into readable text.

Workflow:

1. Load scenario spec from scenario_specs.py.
2. Validate flat observation length against expected host-row layout.
3. Decode per-host slices:
   - subnet one-hot
   - host one-hot
   - compromised/reachable/discovered flags
   - scalar value/discovery_value/access
   - OS/service/process multi-hot blocks
4. Produce a text report with host details and aggregate counts.

Supported scenario names are defined in SCENARIO_SPECS.

## Convergence Callback (callbacks.py)

ConvergenceCallback tracks episode return and convergence speed.

Convergence condition:

- For non-truncated episodes, compute rolling mean episode length over window_size.
- If mean <= ideal_steps + margin for the first time, record convergence_timestep and convergence_episode.

If ideal_steps is missing for a scenario, trainer prints a warning and disables convergence threshold detection for that run.

## Practical Usage

### Dynamic scenario, PPO, masking enabled

```bash
python -m lgrl_final.main \
	--agent-type ppo \
	--scenario-name tiny-gen
```

### Dynamic scenario, PPO, masking disabled (ablation)

```bash
python -m lgrl_final.main \
	--agent-type ppo \
	--scenario-name tiny-gen \
	--disable-action-masking
```

### Static scenario file

```bash
python -m lgrl_final.main \
	--agent-type ppo \
	--scenario-path database/scenarios/medium-multi-site.yml
```

### LGRL run with explicit seeds and JSON summary

```bash
python -m lgrl_final.main \
	--agent-type lgrl \
	--scenario-name tiny-gen \
	--train-seeds 0,1,2,3 \
	--eval-seeds 1000,1001 \
	--json
```

## Python API Usage (for custom LLM client)

CLI currently does not inject a custom LLM client object. Use the Python API when you want real LLM-driven evaluation behavior.

```python
from lgrl_final.main import ExperimentConfig, ExperimentRunner
from lgrl_final.ppo_trainer import PPOTrainer
from lgrl_final.observation_translator import ObservationTranslator


class MyLLMClient:
		def invoke(self, prompt: str):
				# Return any object/string compatible with LLMSubgoalManager parsing.
				return "EXPLOIT_ACCESS"


config = ExperimentConfig(
		agent_type=PPOTrainer.AGENT_TYPE_LGRL,
		scenario_name="tiny-gen",
		llm_client=MyLLMClient(),
		translator=ObservationTranslator(scenario="tiny-gen"),
)

result = ExperimentRunner(config).run()
print(result["metrics"])
```

## Known Caveats

1. Scenario naming mismatch risk:
   - IDEAL_STEPS keys include values like tiny, medium, etc.
   - Generated scenario specs use keys like tiny-gen, medium-gen.
   - If the resolved scenario key is not in IDEAL_STEPS, convergence threshold detection is disabled for that run.
2. evaluate(...) returns only final episode status tuple; aggregate evaluation metrics are exposed via last_eval_metrics.
3. Action mask fallback is intentionally permissive (all true) when no valid action is found, preventing invalid-action deadlocks at the cost of stricter filtering.
4. LLM token usage depends on client response shape and may be zero if usage metadata is absent.

## File-by-File Reference

- main.py: CLI parser, ExperimentConfig, ExperimentRunner.
- ppo_trainer.py: PPOTrainer train/eval/load/save and metric bookkeeping.
- env_factory.py: environment constructors, wrapper stack assembly, reseedable env.
- wrappers.py: action normalization and subgoal wrappers.
- action_mask.py: CustomActionMask validity logic.
- subgoal_manager.py: deterministic and LLM-guided subgoal managers.
- callbacks.py: convergence callback.
- observation_translator.py: flat observation decoding to text.
- scenario_specs.py: scenario metadata table.

### `_host_row_size(cfg)`
