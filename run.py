
#############################################################################
# Run demo of PenGym functionality
#############################################################################

import time
import pengym
import numpy
import logging
import sys
import getopt
import pengym.utilities as utils
from pengym.storyboard import Storyboard
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
from nasim.envs.utils import AccessLevel

storyboard = Storyboard()

#############################################################################
# Wrappers
#############################################################################

class IntActionWrapper(gym.ActionWrapper):
    def action(self, action):
        print(f"Original action: {action}")
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.item())
        if isinstance(action, np.integer):
            return int(action)
        
        print(f"Converted action: {action}")
        return action


#############################################################################
# LGRL Constants & Methods
#############################################################################

# Define subgoals and mapping to indices (SUBGOAL_TO_IDX = {"DISCOVER_HOST": 0, "ENUM_SERVICE": 1, "EXPLOIT_ACCESS": 2, "PRIV_ESC": 3})
SUBGOALS = [
    "DISCOVER_HOST",
    "ENUM_SERVICE",
    "EXPLOIT_ACCESS",
    "PRIV_ESC"
]

SUBGOAL_TO_IDX = {g: i for i, g in enumerate(SUBGOALS)}
NUM_SUBGOALS = len(SUBGOALS)

# Function to convert subgoal name to one-hot vector (vec = DISCOVER_HOST -> [1, 0, 0, 0], ENUM_SERVICE -> [0, 1, 0, 0], etc.)
def one_hot_subgoal(subgoal):
    vec = np.zeros(NUM_SUBGOALS, dtype=np.float32)
    vec[SUBGOAL_TO_IDX[subgoal]] = 1.0
    return vec


# LGRL Subgoal Manager (dummy implementation that updates subgoal based on changes in state counts of known hosts/services/shells)
class SubgoalManager:
    def __init__(self):
        self.reset()

    def reset(self):
        "Reset subgoal manager at the beginning of each episode (current subgoal = DISCOVER_HOST, just_completed = False, prev_counts = 0 for all counts)"
        self.current_subgoal = "DISCOVER_HOST"
        self.just_completed = False
        self.prev_counts = {
            "hosts": 0,
            "services": 0,
            "user_shells": 0,
            "root_shells": 0
        }

    # Update subgoal based on changes in state counts (e.g., if number of known hosts increases, move from DISCOVER_HOST to ENUM_SERVICE, etc.)
    def update(self):
        self.just_completed = False

        curr_hosts = len(utils.host_is_discovered)
        curr_services = 0
        curr_user_shells = 0
        curr_root_shells = 0

        if hasattr(utils, "current_state") and utils.current_state is not None:
            state = utils.current_state

            for host_addr in state.host_num_map.keys():
                host_vec = state.get_host(host_addr)
                
                if host_vec.discovered:
                    curr_hosts += 1
                
                if host_vec.access >= AccessLevel.USER:
                    curr_user_shells += 1
                
                if host_vec.access >= AccessLevel.ROOT:
                    curr_root_shells += 1
            
            curr_services = curr_hosts

        elif hasattr(utils, "host_map"):
            for host_id, host_data in utils.host_map.items():
                if host_data[storyboard.SERVICES] is not None:
                    curr_services += 1
                if host_data[storyboard.SHELL] is not None:
                    curr_user_shells += 1
                if host_data[storyboard.PE_SHELL] is not None:
                    curr_root_shells += 1

            if hasattr(utils, "host_is_discovered"):
                curr_hosts = len(utils.host_is_discovered)

        counts = {
            "hosts": curr_hosts,
            "services": curr_services,
            "user_shells": curr_user_shells,
            "root_shells": curr_root_shells
        }

        if self.current_subgoal == "DISCOVER_HOST":
            if counts["hosts"] > self.prev_counts["hosts"]:
                self.current_subgoal = "ENUM_SERVICE"
                self.just_completed = True

        elif self.current_subgoal == "ENUM_SERVICE":
            if counts["services"] > self.prev_counts["services"]:
                self.current_subgoal = "EXPLOIT_ACCESS"
                self.just_completed = True

        elif self.current_subgoal == "EXPLOIT_ACCESS":
            if counts["user_shells"] > self.prev_counts["user_shells"]:
                self.current_subgoal = "PRIV_ESC"
                self.just_completed = True

        elif self.current_subgoal == "PRIV_ESC":
            if counts["root_shells"] > self.prev_counts["root_shells"]:
                if counts["hosts"] > counts["services"]:
                    self.current_subgoal = "ENUM_SERVICE"
                else:
                    self.current_subgoal = "DISCOVER_HOST"
                self.just_completed = True

        self.prev_counts = counts

    # Get the current subgoal
    def get(self):
        print(f"Current subgoal: {self.current_subgoal}, Just completed: {self.just_completed}, Counts: {self.prev_counts}")
        return self.current_subgoal

# LGRL Subgoal Wrapper that adds the current subgoal as a one-hot vector to the observation (obs = [original_obs, g_vec], where g_vec is the one-hot vector for the current subgoal)
class SubgoalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, subgoal_manager):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager

        # Dimension of the original observation space (e.g., if original obs is a vector of length 10, obs_dim = 10)
        obs_dim = env.observation_space.shape[0]

        # Create a new Box with low=0, high=100, and shape=(obs_dim + NUM_SUBGOALS,) to accommodate the original observation and the one-hot subgoal vector
        # The low and high values can be adjusted based on the expected range of the original observations; here we use 0 and 100 as placeholders
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(obs_dim + NUM_SUBGOALS,),
            dtype=np.float32
        )

    # Update subgoal manager based on the new state after each action and then return the augmented observation
    def observation(self, obs):
        g = self.subgoal_manager.get()
        g_vec = one_hot_subgoal(g)
        return np.concatenate([obs.astype(np.float32), g_vec])
    
# LGRL Reward Wrapper that adds an intrinsic reward (lambda_) when a subgoal is just completed (i.e., when the subgoal manager indicates that the current subgoal was just completed)
class SubgoalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, subgoal_manager, lambda_=0.5):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager
        self.lambda_ = lambda_

    def reward(self, reward):
        if self.subgoal_manager.just_completed:
            return reward + self.lambda_
        return reward

# LGRL Subgoal Update Wrapper that updates the subgoal manager after each action (i.e., after each step, it calls subgoal_manager.update(state) to check if the subgoal should be updated based on the new state)
class SubgoalUpdateWrapper(gym.Wrapper):
    def __init__(self, env, subgoal_manager):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.subgoal_manager.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.subgoal_manager.update()
        return obs, reward, done, truncated, info

#############################################################################
# Deterministic Constants
#############################################################################

# Action names/targets
SUBNET_SCAN = 'SubnetScan'
OS_SCAN = 'OSScan'
SERVICE_SCAN = 'ServiceScan'
EXPLOIT_SSH = 'Exploit_Ssh'
EXPLOIT_FTP = 'Exploit_Ftp'
EXPLOIT_SAMBA = 'Exploit_Samba'
EXPLOIT_SMTP = 'Exploit_Smtp'
EXPLOIT_HTTP = 'Exploit_Http'
PROCESS_SCAN = 'ProcessScan'
PRIVI_ESCA_TOMCAT = 'PrivilegeEscalation_Tomcat'
PRIVI_ESCA_PROFTPD = 'PrivilegeEscalation_Proftpd'
PRIVI_ESCA_CRON = 'PrivilegeEscalation_Cron'

ACTION_NAMES = {SUBNET_SCAN: "subnet_scan", OS_SCAN: "os_scan", SERVICE_SCAN: "service_scan", PROCESS_SCAN: "process_scan",
                EXPLOIT_SSH: "e_ssh",  EXPLOIT_FTP: "e_ftp", EXPLOIT_SAMBA: "e_samba", EXPLOIT_SMTP: "e_smtp", EXPLOIT_HTTP: "e_http", 
                PRIVI_ESCA_TOMCAT: "pe_tomcat", PRIVI_ESCA_PROFTPD: "pe_daclsvc", PRIVI_ESCA_CRON: "pe_schtask"}

HOST1_0 = 'host1-0'
HOST1_1 = 'host1-1'
HOST1_2 = 'host1-2'
HOST1_3 = 'host1-3'
HOST1_4 = 'host1-4'
HOST1_5 = 'host1-5'
HOST1_6 = 'host1-6'
HOST1_7 = 'host1-7'
HOST1_8 = 'host1-8'
HOST1_9 = 'host1-9'
HOST1_10 = 'host1-10'
HOST1_11 = 'host1-11'
HOST1_12 = 'host1-12'
HOST1_13 = 'host1-13'
HOST1_14 = 'host1-14'
HOST1_15 = 'host1-15'
HOST2_0 = 'host2-0'
HOST2_1 = 'host2-1'
HOST3_0 = 'host3-0'
HOST3_1 = 'host3-1'
HOST3_2 = 'host3-2'
HOST3_3 = 'host3-3'
HOST3_4 = 'host3-4'
HOST3_5 = 'host3-5'
HOST4_0 = 'host4-0'
HOST4_1 = 'host4-1'
HOST4_2 = 'host4-2'
HOST4_3 = 'host4-3'
HOST4_4 = 'host4-4'
HOST5_0 = 'host5-0'
HOST5_1 = 'host5-1'
HOST5_2 = 'host5-2'
HOST5_3 = 'host5-3'
HOST6_0 = 'host6-0'
HOST6_1 = 'host6-1'

ACTION_TARGETS = {HOST1_0: (1, 0), HOST1_1: (1, 1), HOST1_2: (1, 2), HOST1_3: (1, 3), HOST1_4: (1, 4), HOST1_5: (1, 5), HOST1_6: (1, 6), HOST1_7: (1, 7), 
                  HOST1_8: (1, 8), HOST1_9: (1, 9), HOST1_10: (1, 10), HOST1_11: (1, 11), HOST1_12: (1, 12), HOST1_13: (1, 13), HOST1_14: (1, 14), HOST1_15: (1, 15),
                  HOST2_0: (2, 0), HOST2_1: (2, 1), 
                  HOST3_0: (3, 0), HOST3_1: (3, 1), HOST3_2: (3, 3), HOST3_3: (3, 3), HOST3_4: (3, 4), HOST3_5: (3, 5), 
                  HOST4_0: (4, 0), HOST4_1: (4, 1),
                  HOST5_0: (5, 0), HOST5_1: (5, 1),
                  HOST6_0: (6, 0), HOST6_1: (6, 1)}

# Agent types
AGENT_TYPE_RANDOM = "random"
AGENT_TYPE_DETERMINISTIC = "deterministic"
AGENT_TYPE_PPO = "ppo"
AGENT_TYPE_LGRL = "lgrl"
CUSTOM_SCRIPT = "check"
DEFAULT_AGENT_TYPE = AGENT_TYPE_DETERMINISTIC

# Other constants
MAX_STEPS = 150 # Max number of pentesting steps (sys.maxsize to disable)
RENDER_OBS_STATE = False

#############################################################################
# Functions
#############################################################################

# Select an action from the action space based on its name
# 'action_name' and its target 'action_target'
def select_action(action_space, action_name, action_target):
    for i in range(0, action_space.n):
        action = action_space.get_action(i)
        if action.name == action_name and action.target == action_target:
            return action

#############################################################################
# Run pentesting with a random agent in the environment 'env'
def run_random_agent(env):

    # Initialize variables
    done = False # Indicate that execution is done
    truncated = False # Indicate that execution is truncated
    step_count = 0 # Count the number of execution steps

    # Loop while the experiment is not finished (pentesting goal not reached)
    # and not truncated (aborted because of exceeding maximum number of steps)
    while not done and not truncated:

        # Sample a random action from the action space of this environment
        action = env.action_space.sample()

        # Increment step count and execute action
        step_count = step_count + 1
        print(f"- Step {step_count}: {env.action_space.get_action(action)}")
        observation, reward, done, truncated, info = env.step(action)
        if RENDER_OBS_STATE:
            env.render() # render most recent observation
            env.render_state() # render most recent state

        # Conditional exit (for debugging purposes)
        if step_count >= MAX_STEPS:
            logging.warning(f"Abort execution after {step_count} steps")
            break

    return done, truncated, step_count

#############################################################################
# Run pentesting with a deterministic agent in the environment 'env'
def run_deterministic_agent(env, deterministic_path):

    # Initialize variables
    done = False # Indicate that execution is done
    truncated = False # Indicate that execution is truncated
    step_count = 0 # Count the number of execution steps

    # Loop while the experiment is not finished (pentesting goal not reached)
    # and not truncated (aborted because of exceeding maximum number of steps)
    while not done and not truncated:
        # Exit if there are no more steps in the deterministic path
        if step_count >= len(deterministic_path):
            break

        # Retrieve the next action to be executed
        action_tuple = deterministic_path[step_count]
        action = select_action(env.action_space, ACTION_NAMES[action_tuple[1]], ACTION_TARGETS[action_tuple[0]])

        # Increment step count and execute action
        step_count = step_count + 1
        
        print(f"- Step {step_count}: {action}")
          
        observation, reward, done, truncated, info = env.step(action)

        if RENDER_OBS_STATE:
            env.render() # render most recent observation
            env.render_state() # render most recent state

        # Conditional exit (for debugging purposes)
        if step_count >= MAX_STEPS:
            logging.warning(f"Abort execution after {step_count} steps")
            break

    return done, truncated, step_count

#############################################################################
# Run pentesting with a Vanilla PPO agent for the specific scenario in 'scenario_path'
def run_ppo_agent(scenario_path):
    def make_env():
        env = create_pengym_custom_environment(scenario_path)
        env = IntActionWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    print("========================================")
    print(vec_env.action_space)
    print(vec_env.action_space.n)
    print(vec_env.observation_space)
    print("========================================")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    print("=================STARTING TRAINING=================")
    model.learn(total_timesteps=100000)
    model.save("ppo_pengym_nasim")
    print("=================TRAINING COMPLETE=================")

    # ===== EVALUATION PHASE =====
    print("\n---------------------------------------")
    print("Evaluation phase:")
    print("---------------------------------------")

    # obs = vec_env.reset()
    eval_env = DummyVecEnv([make_env])
    obs = eval_env.reset()

    done = False # Indicate that execution is done
    truncated = False # Indicate that execution is truncated
    step_count = 0 # Count the number of execution steps
    total_reward = 0 # Accumulate total reward

    while not done and not truncated and step_count < MAX_STEPS:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Get action object for readable output
        action_idx = int(action[0])
        action_obj = eval_env.get_attr('action_space')[0].get_action(action_idx)
        
        print(f"- Step {step_count + 1}: {action_obj}")
        
        # Execute action
        obs, rewards, dones, infos = eval_env.step(action)
        
        step_count += 1
        total_reward += rewards[0]
        done = dones[0]
        
        # Check for truncation (TimeLimit wrapper)
        if 'TimeLimit.truncated' in infos[0]:
            truncated = infos[0]['TimeLimit.truncated']
        
        print(f"  Reward: {rewards[0]:.2f}, Cumulative: {total_reward:.2f}")
        
        # Break if episode ends
        if done or truncated:
            break

    print(f"\nEvaluation complete:")
    print(f"  - Steps: {step_count}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Goal reached: {done}")
    print(f"  - Truncated: {truncated}")

    return done, truncated, step_count

#############################################################################
# Run pentesting with a LGRL agent for the specific scenario in 'scenario_path'
def run_lgrl_agent(scenario_path):
    subgoal_manager = SubgoalManager()

    def make_env():
        env = create_pengym_custom_environment(scenario_path)
        env = IntActionWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

        env = SubgoalUpdateWrapper(env, subgoal_manager)
        env = SubgoalObsWrapper(env, subgoal_manager)
        env = SubgoalRewardWrapper(env, subgoal_manager, lambda_=0.5)

        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    print("========================================")
    print(vec_env.action_space)
    print(vec_env.action_space.n)
    print(vec_env.observation_space)
    print("========================================")

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    print("=================STARTING TRAINING=================")
    model.learn(total_timesteps=100000)
    model.save("ppo_pengym_nasim")
    print("=================TRAINING COMPLETE=================")

    # ===== EVALUATION PHASE =====
    print("\n---------------------------------------")
    print("Evaluation phase:")
    print("---------------------------------------")

    # obs = vec_env.reset()
    eval_env = DummyVecEnv([make_env])
    obs = eval_env.reset()

    done = False # Indicate that execution is done
    truncated = False # Indicate that execution is truncated
    step_count = 0 # Count the number of execution steps
    total_reward = 0 # Accumulate total reward

    while not done and not truncated and step_count < MAX_STEPS:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Get action object for readable output
        action_idx = int(action[0])
        action_obj = eval_env.get_attr('action_space')[0].get_action(action_idx)
        
        print(f"- Step {step_count + 1}: {action_obj}")
        
        # Execute action
        obs, rewards, dones, infos = eval_env.step(action)
        
        step_count += 1
        total_reward += rewards[0]
        done = dones[0]
        
        # Check for truncation (TimeLimit wrapper)
        if 'TimeLimit.truncated' in infos[0]:
            truncated = infos[0]['TimeLimit.truncated']
        
        print(f"  Reward: {rewards[0]:.2f}, Cumulative: {total_reward:.2f}")
        
        # Break if episode ends
        if done or truncated:
            break

    print(f"\nEvaluation complete:")
    print(f"  - Steps: {step_count}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Goal reached: {done}")
    print(f"  - Truncated: {truncated}")

    return done, truncated, step_count

#############################################################################
# Run specific script to check PenGym environment setup'
def run_check(scenario_path):
    def make_env():
        env = create_pengym_custom_environment(scenario_path)
        env = IntActionWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    print("========================================")
    print("Action Space: ", vec_env.action_space)
    print("Action Space Size: ", vec_env.action_space.n)
    print("Observation Space: ", vec_env.observation_space)
    print("Observation Space Shape: ", vec_env.observation_space.shape)
    print("Observation SPace Low: ", vec_env.observation_space.low)
    print("Observation Space High: ", vec_env.observation_space.high)
    print("========================================")

# Create PenGym environment using scenario 'scenario_name'
def create_pengym_environment(scenario_name):
    env = pengym.create_environment(scenario_name)

    # Initialize seed for numpy (used to determine exploit success/failure) and
    # for the environment action space (used to determine order of random actions)
    seed = 1 # NORMAL: No e_ssh failure during pentesting path
    #seed = 300 # INCOMPLETE: Cause e_ssh failure during pentesting path
    numpy.random.seed(seed)
    env.action_space.seed(1)

    return env

# Create PenGym environment using custom scenario
def create_pengym_custom_environment(scenario_path):
    env = pengym.load(scenario_path)

    seed = 1
    numpy.random.seed(seed)
    env.action_space.seed(1)

    return env

# Print usage information
def usage():
    print("\nOVERVIEW: Run demo of the PenGym training framework for pentesting agents\n")
    print("USAGE: python3 run.py [options] <CONFIG_FILE> \n")
    print("OPTIONS:")
    print("-h, --help                     Display this help message and exit")
    print("-a, --agent_type <AGENT_TYPE>  Agent type (random/deterministic)")
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

    # Parse command line arguments
    try:
        # Make sure to add ':' for short-form and '=' for long-form options that require an argument
        opts, trailing_args = getopt.getopt(args, "ha:dn",
                                            ["help", "agent_type=", "disable_pengym", "nasim_simulation"])
    except getopt.GetoptError as err:
        logging.error(f"Command-line argument error: {str(err)}")
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-a", "--agent"):
            agent_type = arg
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
    print(f"* Create environment using custom scenario from '{scenario_path}'...")
    env = create_pengym_custom_environment(scenario_path)

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

    # Run experiment using a random agent
    if agent_type == AGENT_TYPE_RANDOM:
        print("* Perform pentesting using a RANDOM agent...")
        done, truncated, step_count = run_random_agent(env)

    # Run experiment using a PPO agent
    elif agent_type == AGENT_TYPE_PPO:
        print("* Perform pentesting using a PPO agent...")
        done, truncated, step_count = run_ppo_agent(scenario_path)
    
    # Run experiment using a LGRL agent
    elif agent_type == AGENT_TYPE_LGRL:
        print("* Perform pentesting using a LGRL agent...")
        done, truncated, step_count = run_lgrl_agent(scenario_path)
    
    # Run a custom script to check environment setup
    elif agent_type == CUSTOM_SCRIPT:
        print("* Perform check using custom script...")
        run_check(scenario_path)
        exit(0)

    # Run experiment using a deterministic agent
    elif agent_type == AGENT_TYPE_DETERMINISTIC:

        # Set up deterministic path

        # Optimal path for scenario "medium-single-site" according to "medium-single-site.yaml" note
        deterministic_path = [ (HOST5_1, EXPLOIT_HTTP), (HOST5_1, SUBNET_SCAN),
                        (HOST2_1, EXPLOIT_SMTP), (HOST2_1, SUBNET_SCAN),
                        (HOST3_1, EXPLOIT_HTTP),
                        (HOST3_4, EXPLOIT_SSH), (HOST3_4, PRIVI_ESCA_TOMCAT)]

        # Pentesting path for scenario "tiny" assuming need for service/process discovery
        deterministic_path = [ (HOST5_1, OS_SCAN), (HOST5_1, SERVICE_SCAN), (HOST5_1, EXPLOIT_HTTP), (HOST5_1, SUBNET_SCAN),
                (HOST2_1, OS_SCAN), (HOST2_1, SERVICE_SCAN), (HOST2_1, EXPLOIT_SMTP), (HOST2_1, SUBNET_SCAN),
                (HOST3_1, OS_SCAN), (HOST3_1, SERVICE_SCAN), (HOST3_1, EXPLOIT_HTTP),
                (HOST3_4, OS_SCAN), (HOST3_4, SERVICE_SCAN), (HOST3_4, EXPLOIT_SSH), (HOST3_4, PROCESS_SCAN), (HOST3_4, PRIVI_ESCA_TOMCAT)]

        print("* Execute pentesting using a DETERMINISTIC agent...")
        done, truncated, step_count = run_deterministic_agent(env, deterministic_path)

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
    #print(f"Execution Time: {end-start:1.6f}s")
