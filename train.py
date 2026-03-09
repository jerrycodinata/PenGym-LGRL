import time
from os.path import isfile
import logging
import sys
import getopt
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from sb3_contrib import MaskablePPO
import pengym.utilities as utils
from pengym.storyboard import Storyboard
from lgrl.subgoal_managers import OracleSubgoalManager
from lgrl.common_utilities import make_env
from lgrl.action_mask import custom_mask_fn
from lgrl.evaluation import ConvergenceCallback

storyboard = Storyboard()

# Agent types
AGENT_TYPE_PPO = "ppo"
AGENT_TYPE_LGRL = "lgrl"
DEFAULT_AGENT_TYPE = AGENT_TYPE_PPO

# Other constants
MAX_STEPS = 150 # Max number of pentesting steps (sys.maxsize to disable)
TOTAL_EPISODES = 500
EVAL_EPISODES = 100
TOTAL_TIMESTEPS = MAX_STEPS * TOTAL_EPISODES
FRAME_MEMORY = 4 # Number of frames to stack for PPO agent
WINDOW_SIZE = 5 # Number of episodes to consider for convergence calculation
MARGIN = 2 # Margin of steps above ideal steps to consider as convergence
RENDER_OBS_STATE = False

IDEAL_STEPS = {
    "tiny": 6,
    "tiny-small": 7,
    "tiny-hard": 5,
    "small-linear": 12,
    "small-honeypot": 8,
    "medium": 8,
    "medium-single-site": 4,
    "medium-multi-site": 7,
}

def run_agent(agent_type, scenario_path, model_path=None):
    scenario_name = scenario_path.split("/")[-1].split(".")[0]
    subgoal_manager = None
    env_fn = None
    model = None
    convergence_speed = -1

    if agent_type == AGENT_TYPE_LGRL:
        subgoal_manager = OracleSubgoalManager(utils=utils, storyboard=storyboard)
        env_fn = lambda: make_env(
            scenario_path, 
            max_episode_steps=MAX_STEPS, 
            llm_guidance=True, 
            subgoal_manager=subgoal_manager, 
            intrinsic_reward=False, 
            intrinsic_reward_lambda=10)
    elif agent_type == AGENT_TYPE_PPO:
        env_fn = lambda: make_env(
            scenario_path, 
            max_episode_steps=MAX_STEPS)

    if model_path is not None:
        model = MaskablePPO.load(model_path)
    else:
        vec_env = DummyVecEnv([env_fn])
        vec_env = VecFrameStack(vec_env, n_stack=FRAME_MEMORY)

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

        ideal_step = IDEAL_STEPS.get(scenario_name)
        convergence_cb = ConvergenceCallback(ideal_steps=ideal_step, window_size=WINDOW_SIZE, margin=MARGIN)

        print("=================STARTING TRAINING=================")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=convergence_cb)

        # Save Model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        scenario_name = scenario_path.split("/")[-1].split(".")[0]
        model_name = f"{agent_type}_{scenario_name}_({MAX_STEPS}_{TOTAL_EPISODES}ep)_{timestamp}"
        model.save(f"models/{model_name}")

        # Store CS metric for evaluation summary
        convergence_speed = convergence_cb.convergence_timestep
        print("=================TRAINING COMPLETE=================")

    # ===== EVALUATION PHASE =====
    print("\n---------------------------------------")
    print("Evaluation phase:")
    print("---------------------------------------")

    eval_vec_env = DummyVecEnv([env_fn])
    eval_vec_env = VecFrameStack(eval_vec_env, n_stack=FRAME_MEMORY)

    success_count = 0
    total_tokens_used = 0
    total_cumulative_reward = 0

    for ep in range(EVAL_EPISODES):
        print(f"\n=== Evaluation Episode {ep + 1}/{EVAL_EPISODES} ===")

        obs = eval_vec_env.reset()

        terminated = False
        truncated = False
        ep_steps = 0
        ep_reward = 0
        ep_token_usage = 0

        while not terminated and not truncated and ep_steps < MAX_STEPS:
            action_masks = np.array([custom_mask_fn(eval_vec_env.envs[0])])
            
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

            base_env = eval_vec_env.envs[0].env.unwrapped 
            action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            action_obj = base_env.action_space.get_action(action_idx)
            print(f"- Step {ep_steps + 1}: {action_obj}")
            
            obs, reward, done, info = eval_vec_env.step(action)
            terminated = done[0]
            truncated = info[0].get('TimeLimit.truncated', False)
            ep_steps += 1
            ep_reward += reward[0]

            if RENDER_OBS_STATE:
                base_env.render()
                base_env.render_state()

            print(f"  Reward: {reward[0]:.2f}, Cumulative: {ep_reward:.2f}")

        done = bool(terminated and not truncated)
        success_count += 1 if done else 0
        total_cumulative_reward += ep_reward
        total_tokens_used += ep_token_usage

        print(f"\nEvaluation {ep + 1}/{EVAL_EPISODES} complete:")
        print(f"  - Steps: {ep_steps}")
        print(f"  - Total reward: {ep_reward:.2f}")
        print(f"  - Goal reached: {done}")
        print(f"  - Truncated: {truncated}")
    
    # Final Metrics Calculation
    success_rate = (success_count / EVAL_EPISODES) * 100
    avg_cumulative_reward = total_cumulative_reward / EVAL_EPISODES
    avg_tokens_used = total_tokens_used / EVAL_EPISODES

    print("\n=======================================")
    print("Evaluation Summary:")
    print("=======================================")
    print(f"Agent Type                 : {agent_type.upper()}")
    print(f"Total Evaluation Episodes  : {EVAL_EPISODES}")
    print(f"Convergence Speed          : {convergence_speed} timesteps") 
    print(f"Success Rate               : {success_rate:.2f}%")
    print(f"Average Cumulative Reward  : {avg_cumulative_reward:.2f}")
    print(f"Average Tokens Used        : {avg_tokens_used:.2f}")

    return done, truncated, ep_steps

# Print usage information
def usage():
    print("\nOVERVIEW: Run demo of the PenGym training framework for pentesting agents\n")
    print("USAGE: python3 run.py [options] <CONFIG_FILE> \n")
    print("OPTIONS:")
    print("-h, --help                     Display this help message and exit")
    print("-a, --agent_type <AGENT_TYPE>  Agent type (ppo/lgrl)")
    print("-l, --load_model <MODEL_PATH>  Load a pre-trained model from the specified path")
    print("-d, --disable_pengym           Disable PenGym execution in cyber range")
    print("-n, --nasim_simulation         Enable NASim simulation execution")

#############################################################################
# Main program
#############################################################################
def main(args):

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='* %(levelname)s: %(filename)s: %(message)s')


    print("#########################################################################")
    print("PenGym: Pentesting Training Framework for Reinforcement Learning Agents")
    print("#########################################################################")

    # Default argument values
    agent_type = DEFAULT_AGENT_TYPE
    config_path = None
    model_path = None

    # Parse command line arguments
    try:
        # Make sure to add ':' for short-form and '=' for long-form options that require an argument
        opts, trailing_args = getopt.getopt(args, "ha:l:dn",
                                            ["help", "agent_type=", "load_model=", "disable_pengym", "nasim_simulation"])
    except getopt.GetoptError as err:
        logging.error(f"Command-line argument error: {str(err)}")
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-a", "--agent_type"):
            agent_type = arg
        elif opt in ("-l", "--load_model"):
            model_path = arg
        elif opt in ("-d", "--disable_pengym"):
            utils.ENABLE_PENGYM = False
        elif opt in ("-n", "--nasim_simulation"):
            utils.ENABLE_NASIM = True
        else:
            # Nothing to do, since unrecognized options are caught by
            # the getopt.GetoptError exception above
            pass

    # Get path of configuration file
    try:
        config_path = trailing_args[0]
    except Exception as e:
        logging.error(f"Configuration file is not specified")
        usage()
        sys.exit(2)
    
    if model_path is not None:
        filename = model_path + ".zip"

        if isfile(filename):
            print(f"* Load pre-trained model from '{filename}'...")
        else:
            logging.error("Specified model path does not exist")
            sys.exit(2)

    # Print parameters
    print(f"* Execution parameters:")
    print(f"  - Agent type: {agent_type}")
    print(f"  - PenGym cyber range execution enabled: {utils.ENABLE_PENGYM}")
    print(f"  - NASim simulation execution enabled: {utils.ENABLE_NASIM}")

    # Check execution parameters
    if not (utils.ENABLE_PENGYM or utils.ENABLE_NASIM):
        logging.error("Either PenGym or NASim must be enabled")
        usage()
        sys.exit(2)

    print(f"* Read configuration from '{config_path}'...")
    utils.init_config_info(config_path)

    # Create an experiment environment using scenario path
    scenario_path = utils.replace_file_path(utils.config_info, storyboard.SCENARIO_FILE)

    if utils.ENABLE_PENGYM:
        print(f"* Read configuration from '{config_path}'...")
        utils.init_config_info(config_path)
        
        print("* Initialize MSF RPC client...")
        utils.init_msfrpc_client()
        
        print("* Initialize Nmap Scanner...")
        utils.init_nmap_scanner()
        
        # Create host map dictionary
        range_detail_file = utils.replace_file_path(database=utils.config_info,
                                                     file_name=storyboard.RANGE_DETAILS_FILE)

        utils.init_host_map(range_details_file=range_detail_file)

        # Initializer map of service ports
        utils.init_service_port_map()
    
        # Deactivate bridge that not conneccted to Internet
        utils.init_bridge_setup(range_details_file=range_detail_file)

    # Run experiment using a PPO agent
    if agent_type == AGENT_TYPE_PPO:
        print("* Perform pentesting using a PPO agent...")
        if model_path is not None:
            print(f"* Load pre-trained model from '{model_path}.zip'...")
            done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_path, model_path=model_path)
        else:
            print("* Training from scratch...")
            done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_path)
    
    # Run experiment using a LGRL agent
    elif agent_type == AGENT_TYPE_LGRL:
        print("* Perform pentesting using a LGRL agent...")
        done, truncated, step_count = run_agent(AGENT_TYPE_LGRL, scenario_path)

    else:
        logging.error(f"Unrecognized agent type: '{agent_type}'")
        usage()
        sys.exit(1)

    # Print execution status
    if done:
        # All the goals in the scenario file were reached
        print(f"* NORMAL execution: {step_count} steps")
    elif truncated:
        # Execution was truncated before reaching all the goals (for random agents, etc.)
        print(f"* TRUNCATED execution: {step_count} steps")
    else:
        # Execution finished before reaching all the goals (for deterministic agents)
        print(f"* INCOMPLETE execution: {step_count} steps")

    if utils.ENABLE_PENGYM:
        print("* Clean up MSF RPC client...")
        utils.cleanup_msfrpc_client()

        print("* Restore the to intial state of the firewalls for all hosts...")
        utils.save_restore_firewall_rules_all_hosts(flag=storyboard.RESTORE)

#############################################################################
# Run program
if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print(f"Execution Time: {end-start:1.6f}s")