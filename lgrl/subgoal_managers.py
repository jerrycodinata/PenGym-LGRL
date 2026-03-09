from nasim.envs.utils import AccessLevel

# Oracle Subgoal Manager
class OracleSubgoalManager:
    utils = None
    storyboard = None

    def __init__(self, utils, storyboard):
        self.utils = utils
        self.storyboard = storyboard
        self.reset()

    def reset(self):
        # The agent starts the episode needing to exploit the initially visible host
        self.current_subgoal = "EXPLOIT_ACCESS" 
        self.just_completed = False
        self.prev_counts = {
            "hosts": 0,
            "user_shells": 0,
            "root_shells": 0
        }

    def update(self):
        self.just_completed = False

        curr_hosts = 0
        curr_user_shells = 0
        curr_root_shells = 0

        # --- Count Extraction ---
        if hasattr(self.utils, "current_state") and self.utils.current_state is not None:
            state = self.utils.current_state
            for host_addr in state.host_num_map.keys():
                host_vec = state.get_host(host_addr)
                if host_vec.discovered:
                    curr_hosts += 1
                if host_vec.access >= AccessLevel.USER:
                    curr_user_shells += 1
                if host_vec.access >= AccessLevel.ROOT:
                    curr_root_shells += 1
                    
        elif hasattr(self.utils, "host_map"):
            for host_id, host_data in self.utils.host_map.items():
                if host_data.get(self.storyboard.SHELL) is not None:
                    curr_user_shells += 1
                if host_data.get(self.storyboard.PE_SHELL):
                    curr_root_shells += 1
            if hasattr(self.utils, "host_is_discovered"):
                curr_hosts = len(self.utils.host_is_discovered)

        # Initialize base counts on the very first step of the episode
        if self.prev_counts["hosts"] == 0:
            self.prev_counts["hosts"] = curr_hosts
            self.prev_counts["user_shells"] = curr_user_shells
            self.prev_counts["root_shells"] = curr_root_shells
            return

        counts = {
            "hosts": curr_hosts,
            "user_shells": curr_user_shells,
            "root_shells": curr_root_shells
        }

        # --- ORACLE STATE MACHINE FOR MEDIUM-MULTI-SITE ---
        if self.current_subgoal == "EXPLOIT_ACCESS":
            if counts["user_shells"] > self.prev_counts["user_shells"]:
                self.just_completed = True
                
                # Logic mapped to optimal path:
                # 1st Shell on (6,1) -> Next step is Subnet Scan
                # 2nd Shell on (2,1) -> Next step is Subnet Scan
                if counts["user_shells"] in [1, 2]:
                    self.current_subgoal = "DISCOVER_HOST"
                
                # 3rd Shell on (3,1) -> Next step is another Exploit on (3,4)
                elif counts["user_shells"] == 3:
                    self.current_subgoal = "EXPLOIT_ACCESS"
                
                # 4th Shell on (3,4) -> Time to Priv Esc
                elif counts["user_shells"] >= 4:
                    self.current_subgoal = "PRIV_ESC"

        elif self.current_subgoal == "DISCOVER_HOST":
            if counts["hosts"] > self.prev_counts["hosts"]:
                self.just_completed = True
                # After discovering a subnet, the optimal path always exploits immediately
                self.current_subgoal = "EXPLOIT_ACCESS"

        elif self.current_subgoal == "PRIV_ESC":
            if counts["root_shells"] > self.prev_counts["root_shells"]:
                self.just_completed = True
                # Path is complete. Keep rewarding if it somehow finds more, or hold state.
                self.current_subgoal = "PRIV_ESC"

        self.prev_counts = counts

    def get(self):
        return self.current_subgoal
