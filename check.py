import time
import logging
import sys
import getopt
import numpy as np
import math
from collections import Counter
import pengym
import pengym.utilities as utils
from pengym.storyboard import Storyboard

storyboard = Storyboard()

# Other constants
MAX_STEPS = 150 # Max number of pentesting steps (sys.maxsize to disable)

def print_observation_details(env, observation):
    flat_obs = np.asarray(observation).reshape(-1)

    print("\nExpanded Observation (flat vector):")
    print(np.array2string(flat_obs, threshold=np.inf, max_line_width=200))

    print("\nExpanded Observation (index -> value):")
    for index, value in enumerate(flat_obs):
        print(f"  [{index:03d}] {value}")

    # last_obs keeps NASim Observation object regardless of flat_obs setting
    host_obs, aux_obs = env.last_obs.get_readable()

    print("\nExpanded Observation (discrete vector):")
    discrete_obs = [math.ceil(v) for v in flat_obs]

    num_hosts = len(host_obs)
    if num_hosts > 0:
        obs_per_host = len(discrete_obs) // (num_hosts + 1)
        for i in range(num_hosts):
            host_start = i * obs_per_host
            host_end = host_start + obs_per_host
            print(f"  Host {i}: {discrete_obs[host_start:host_end]}")
        
        aux_end = len(discrete_obs)
        aux_start = aux_end - obs_per_host
        print(f"  Aux   : {discrete_obs[aux_start:aux_end]}")
    else:
        print(f"{discrete_obs}\n")

    print("\nFlattened Observation:")
    print(discrete_obs)

    print("\nExpanded Observation (decoded by host):")
    for host_index, host_data in enumerate(host_obs):
        print(f"  HostRow {host_index}: {host_data}")

    print("\nExpanded Observation (aux row):")
    print(f"  {aux_obs}")

    # obs = env.reset()

    # obs_2d = obs.reshape(-1, obs.shape[0] // (env.num_hosts + 1))
    # print(obs_2d.shape)

    # for i, val in enumerate(obs):
    #     print(f"{i}: {val}")


def print_action_details(env):
    action_space = env.action_space
    actions = getattr(action_space, "actions", None)

    if actions is None and not hasattr(action_space, "get_action"):
        return

    print("\nAction Space Details:")
    print("---------------------------------------")

    if actions is not None:
        action_types = Counter(action.__class__.__name__ for action in actions)
        print("Action Type Counts:")
        for action_type, count in sorted(action_types.items()):
            print(f"  - {action_type}: {count}")

        print("\nAction Index Mapping:")
        for index, action in enumerate(actions):
            print(f"  [{index:02d}] {action}")

    if hasattr(action_space, "n") and hasattr(action_space, "get_action"):
        print("\nAction Space Used in Train/Test:")
        for i in range(action_space.n):
            action = action_space.get_action(i)
            action_name = getattr(action, "name", None)
            action_target = getattr(action, "target", None)
            if action_name is not None and action_target is not None:
                print(f"  - {i:02d}: {action_name} on target {action_target}")
            else:
                print(f"  - {i:02d}: {action}")
    elif actions is None:
        print("\nAction Space Used in Train/Test:")
        print("  (unavailable: action_space.get_action missing)")

def check_spaces(scenario_path):
    fully_obs = False
    env = pengym.load(scenario_path, fully_obs=fully_obs)
    np.random.seed(1)
    env.action_space.seed(1)
    observation, _ = env.reset(seed=1)

    print("\n---------------------------------------")
    print("Environment Space Check:")
    print("---------------------------------------")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space     : {env.action_space}")
    print(f"Fully Observable : {fully_obs}")

    if hasattr(env.observation_space, "shape"):
        print(f"Observation Shape: {env.observation_space.shape}")
    if hasattr(env.action_space, "n"):
        print(f"Number of Actions: {env.action_space.n}")

    print_observation_details(env, observation)
    print_action_details(env)

    env.close()

# Print usage information
def usage():
    print("\nOVERVIEW: Check PenGym environment observation/action spaces\n")
    print("USAGE: python3 check.py [options] <CONFIG_FILE> \n")
    print("OPTIONS:")
    print("-h, --help                     Display this help message and exit")
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
    print("PenGym: Environment Space Checker")
    print("#########################################################################")

    # Default argument values
    config_path = None

    # Parse command line arguments
    try:
        # Make sure to add ':' for short-form and '=' for long-form options that require an argument
        opts, trailing_args = getopt.getopt(args, "hdn",
                            ["help", "disable_pengym", "nasim_simulation"])
    except getopt.GetoptError as err:
        logging.error(f"Command-line argument error: {str(err)}")
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
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

    # Only check environment spaces
    check_spaces(scenario_path)

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