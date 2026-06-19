#!/bin/bash

python -m lgrl_final.ablation_study --scenario-name small-gen --total-timesteps 200000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name small-gen --total-timesteps 200000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 500000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 500000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 500000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 2000000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 2000000 --llm-model deepseek-v4-pro
python -m lgrl_final.ablation_study --scenario-name medium-gen --total-timesteps 2000000 --llm-model deepseek-v4-pro

