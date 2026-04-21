from abc import ABC, abstractmethod
from typing import Any, Optional

from nasim.envs.utils import AccessLevel


SUBGOALS = [
    "DISCOVER_HOST",
    "ENUM_SERVICE",
    "EXPLOIT_ACCESS",
    "PRIV_ESC",
]

SUBGOAL_TO_IDX = {g: i for i, g in enumerate(SUBGOALS)}


class BaseSubgoalManager(ABC):
    def __init__(self):
        self.current_subgoal = "EXPLOIT_ACCESS"
        self.just_completed = False

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self):
        raise NotImplementedError()

    def get(self) -> str:
        return self.current_subgoal


class DeterministicSubgoalManager(BaseSubgoalManager):
    def __init__(self, utils, storyboard):
        super().__init__()
        self.utils = utils
        self.storyboard = storyboard
        self.prev_counts = {}
        self.reset()

    def reset(self):
        self.current_subgoal = "EXPLOIT_ACCESS"
        self.just_completed = False
        self.prev_counts = {
            "hosts": 0,
            "user_shells": 0,
            "root_shells": 0,
        }

    def _extract_counts(self):
        curr_hosts = 0
        curr_user_shells = 0
        curr_root_shells = 0

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
            for _, host_data in self.utils.host_map.items():
                if host_data.get(self.storyboard.SHELL) is not None:
                    curr_user_shells += 1
                if host_data.get(self.storyboard.PE_SHELL):
                    curr_root_shells += 1

            if hasattr(self.utils, "host_is_discovered"):
                curr_hosts = len(self.utils.host_is_discovered)

        return {
            "hosts": curr_hosts,
            "user_shells": curr_user_shells,
            "root_shells": curr_root_shells,
        }

    def update(self):
        self.just_completed = False
        counts = self._extract_counts()

        if self.prev_counts["hosts"] == 0:
            self.prev_counts = counts
            return

        if self.current_subgoal == "EXPLOIT_ACCESS":
            if counts["user_shells"] > self.prev_counts["user_shells"]:
                self.just_completed = True
                if counts["user_shells"] in [1, 2]:
                    self.current_subgoal = "DISCOVER_HOST"
                elif counts["user_shells"] == 3:
                    self.current_subgoal = "EXPLOIT_ACCESS"
                elif counts["user_shells"] >= 4:
                    self.current_subgoal = "PRIV_ESC"

        elif self.current_subgoal == "DISCOVER_HOST":
            if counts["hosts"] > self.prev_counts["hosts"]:
                self.just_completed = True
                self.current_subgoal = "EXPLOIT_ACCESS"

        elif self.current_subgoal == "PRIV_ESC":
            if counts["root_shells"] > self.prev_counts["root_shells"]:
                self.just_completed = True
                self.current_subgoal = "PRIV_ESC"

        self.prev_counts = counts


class LLMSubgoalManager(BaseSubgoalManager):
    def __init__(
        self,
        utils,
        storyboard,
        llm_client: Optional[Any] = None,
        translator: Optional[Any] = None,
        fallback_manager: Optional[DeterministicSubgoalManager] = None,
    ):
        super().__init__()
        self.utils = utils
        self.storyboard = storyboard
        self.llm_client = llm_client
        self.translator = translator
        self.fallback_manager = fallback_manager or DeterministicSubgoalManager(utils, storyboard)
        self.reset()

    def reset(self):
        self.fallback_manager.reset()
        self.current_subgoal = self.fallback_manager.get()
        self.just_completed = False

    def _parse_subgoal(self, text: str) -> Optional[str]:
        if not text:
            return None
        normalized = text.upper()
        for subgoal in SUBGOALS:
            if subgoal in normalized:
                return subgoal
        return None

    def _build_prompt(self) -> str:
        context = ""
        if self.translator is not None:
            try:
                if hasattr(self.utils, "current_state"):
                    context = self.translator.translate(self.utils.current_state)
                else:
                    context = self.translator.translate()
            except Exception:
                context = ""

        return (
            "Choose exactly one next pentest subgoal from: "
            + ", ".join(SUBGOALS)
            + ". Reply with only one token from the list.\n"
            + f"Current subgoal: {self.current_subgoal}\n"
            + (f"Context:\n{context}" if context else "")
        )

    def _query_llm(self, prompt: str) -> Optional[str]:
        if self.llm_client is None:
            return None

        response = None
        if callable(self.llm_client):
            response = self.llm_client(prompt)
        elif hasattr(self.llm_client, "generate"):
            response = self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, "complete"):
            response = self.llm_client.complete(prompt)
        elif hasattr(self.llm_client, "invoke"):
            response = self.llm_client.invoke(prompt)

        if response is None:
            return None

        if isinstance(response, str):
            return response

        if hasattr(response, "content"):
            return str(response.content)

        return str(response)

    def update(self):
        self.fallback_manager.update()
        self.just_completed = self.fallback_manager.just_completed

        # Keep deterministic behavior during non-transition steps.
        if not self.just_completed:
            self.current_subgoal = self.fallback_manager.get()
            return

        llm_choice = None
        prompt = self._build_prompt()
        raw_response = self._query_llm(prompt)
        if raw_response is not None:
            llm_choice = self._parse_subgoal(raw_response)

        self.current_subgoal = llm_choice or self.fallback_manager.get()
