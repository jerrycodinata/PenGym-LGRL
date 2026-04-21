from typing import Optional

import numpy as np

from lgrl_final.scenario_specs import get_scenario_spec


class ObservationTranslator:
    """Translate flat NASIM observations into text for LLM prompting/debugging."""

    def __init__(self, observation=None, scenario: Optional[str] = None):
        self.observation = observation
        self.scenario = scenario
        self.desc = None

    def update(self, observation):
        self.observation = observation

    def get_detail(self, scenario: Optional[str] = None):
        scenario_name = scenario or self.scenario
        if scenario_name is None:
            raise ValueError("Scenario is not set.")
        return get_scenario_spec(scenario_name)

    def translate(self, observation=None, scenario: Optional[str] = None):
        obs = self.observation if observation is None else observation
        scenario_name = self.scenario if scenario is None else scenario

        if obs is None:
            return "No observation available."
        if scenario_name is None:
            return "Scenario is not set for translation."

        cfg = get_scenario_spec(scenario_name)
        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim != 1:
            return f"Unsupported observation shape: {arr.shape}"

        host_rows = self._reshape_flat_observation(arr, cfg)
        return self._format_text(host_rows, cfg, scenario_name)

    def get_description(self) -> str:
        self.desc = self.translate()
        return self.desc

    @staticmethod
    def _host_row_size(cfg) -> int:
        return (
            cfg["subnets"]
            + cfg["hosts"]
            + 6
            + cfg["os"]
            + cfg["services"]
            + cfg["processes"]
        )

    def _reshape_flat_observation(self, flat_obs: np.ndarray, cfg) -> np.ndarray:
        row_size = self._host_row_size(cfg)
        expected_total = cfg["hosts"] * row_size

        if flat_obs.size != expected_total:
            raise ValueError(
                f"Observation length {flat_obs.size} does not match expected "
                f"length {expected_total} for scenario {self.scenario}."
            )

        return flat_obs.reshape(cfg["hosts"], row_size)

    def _format_text(self, host_rows: np.ndarray, cfg, scenario_name: str) -> str:
        lines = [
            "Observation Summary",
            f"Scenario: {scenario_name}",
            f"Total Host: {cfg['hosts']}",
            f"Total Subnet: {cfg['subnets']}",
            f"OS Types: {cfg['os']}",
            f"Services: {cfg['services']}",
            f"Processes: {cfg['processes']}",
            f"Exploits: {cfg['exploits']}",
            f"Privesc: {cfg['privesc']}",
            f"Actions: {cfg['actions']}",
            f"Dimensions: {cfg['dims_x']}x{cfg['dims_y']}",
            "",
            "Host Details",
        ]

        discovered_count = 0
        compromised_count = 0
        reachable_count = 0

        for idx, row in enumerate(host_rows):
            host_info = self._decode_host_row(row, cfg)
            if host_info["discovered"]:
                discovered_count += 1
            if host_info["compromised"]:
                compromised_count += 1
            if host_info["reachable"]:
                reachable_count += 1

            lines.append(
                f"- Host#{idx + 1}: subnet={host_info['subnet_addr']}, "
                f"host_addr={host_info['host_addr']}, discovered={host_info['discovered']}, "
                f"reachable={host_info['reachable']}, compromised={host_info['compromised']}, "
                f"value={host_info['value']:.2f}, discovery_value={host_info['discovery_value']:.2f}, "
                f"access={host_info['access']}"
            )
            lines.append(
                f"  os={host_info['os']}, services={host_info['services_running']}, "
                f"processes={host_info['processes_running']}"
            )

        lines.append("")
        lines.append(
            "Counts: "
            f"discovered_hosts={discovered_count}, "
            f"reachable_hosts={reachable_count}, "
            f"compromised_hosts={compromised_count}"
        )

        return "\n".join(lines)

    def _decode_host_row(self, row: np.ndarray, cfg) -> dict:
        i = 0

        subnet_slice = row[i : i + cfg["subnets"]]
        i += cfg["subnets"]

        host_slice = row[i : i + cfg["hosts"]]
        i += cfg["hosts"]

        compromised = bool(row[i] > 0.5)
        i += 1
        reachable = bool(row[i] > 0.5)
        i += 1
        discovered = bool(row[i] > 0.5)
        i += 1

        value = float(row[i])
        i += 1
        discovery_value = float(row[i])
        i += 1
        access = self._access_to_label(row[i])
        i += 1

        os_slice = row[i : i + cfg["os"]]
        i += cfg["os"]

        services_slice = row[i : i + cfg["services"]]
        i += cfg["services"]

        processes_slice = row[i : i + cfg["processes"]]

        return {
            "subnet_addr": self._decode_single_onehot(subnet_slice, "subnet"),
            "host_addr": self._decode_single_onehot(host_slice, "host"),
            "compromised": compromised,
            "reachable": reachable,
            "discovered": discovered,
            "value": value,
            "discovery_value": discovery_value,
            "access": access,
            "os": self._decode_multi_hot(os_slice, "os"),
            "services_running": self._decode_multi_hot(services_slice, "service"),
            "processes_running": self._decode_multi_hot(processes_slice, "process"),
        }

    @staticmethod
    def _decode_single_onehot(vec: np.ndarray, prefix: str) -> str:
        active = np.flatnonzero(vec > 0.5)
        if active.size == 0:
            return "unknown"
        if active.size == 1:
            return f"{prefix}_{int(active[0])}"
        return "+".join(f"{prefix}_{int(idx)}" for idx in active)

    @staticmethod
    def _decode_multi_hot(vec: np.ndarray, prefix: str) -> str:
        active = np.flatnonzero(vec > 0.5)
        if active.size == 0:
            return "none"
        return ", ".join(f"{prefix}_{int(idx)}" for idx in active)

    @staticmethod
    def _access_to_label(access: float) -> str:
        level = int(round(float(access)))
        if level <= 0:
            return "NONE"
        if level == 1:
            return "USER"
        return "ROOT"
