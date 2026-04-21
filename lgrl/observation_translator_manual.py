from __future__ import annotations

import numpy as np


class ObservationTranslator:
    """Translate flat NASIM observation vectors into text.

    Expected 1D observation layout per host:
    [subnet_addr_onehot, host_addr_onehot, compromised, reachable, discovered,
    value, discovery_value, access, os_onehot, services_onehot, processes_onehot]
    """

    DICTIONARY_KEY = {
        "tiny-gen": {
            "subnets": 4,
            "hosts": 3,
            "os": 1,
            "services": 1,
            "processes": 1,
            "exploits": 1,
            "privesc": 1,
            "actions": 18,
            "dims_x": 4,
            "dims_y": 14            
        },
        "small-gen": {
            "subnets": 5,
            "hosts": 8,
            "os": 2,
            "services": 3,
            "processes": 2,
            "exploits": 3,
            "privesc": 2,
            "actions": 72,
            "dims_x": 9,
            "dims_y": 14 
        },
        "medium-gen": {
            "subnets": 6,
            "hosts": 16,
            "os": 2,
            "services": 5,
            "processes": 2,
            "exploits": 5,
            "privesc": 2,
            "actions": 176,
            "dims_x": 17,
            "dims_y": 26 
        },
        "large-gen": {
            "subnets": 8,
            "hosts": 23,
            "os": 3,
            "services": 7,
            "processes": 3,
            "exploits": 7,
            "privesc": 3,
            "actions": 322,
            "dims_x": 24,
            "dims_y": 32 
        }
    }

    def __init__(self, observation, scenario):
        self.observation = observation
        self.scenario = scenario
        self.desc = None

    def translate(self):
        obs = self.observation

        if obs is None:
            return "No observation available."

        if self.scenario not in self.DICTIONARY_KEY:
            return f"Unsupported scenario for translation: {self.scenario}"

        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim != 1:
            return f"Unsupported observation shape: {arr.shape}"

        host_rows = self._reshape_flat_observation(arr)
        return self._format_text(host_rows)

    def get_description(self) -> str:
        self.desc = self.translate()
        return self.desc

    def _host_row_size(self) -> int:
        cfg = self.DICTIONARY_KEY[self.scenario]
        return (
            cfg["subnets"]
            + cfg["hosts"]
            + 6
            + cfg["os"]
            + cfg["services"]
            + cfg["processes"]
        )

    def _reshape_flat_observation(self, flat_obs: np.ndarray) -> np.ndarray:
        cfg = self.DICTIONARY_KEY[self.scenario]
        row_size = self._host_row_size()
        expected_total = cfg["hosts"] * row_size

        if flat_obs.size != expected_total:
            raise ValueError(
                f"Observation length {flat_obs.size} does not match expected "
                f"length {expected_total} for scenario {self.scenario}."
            )

        return flat_obs.reshape(cfg["hosts"], row_size)

    def _format_text(self, host_rows: np.ndarray) -> str:
        cfg = self.DICTIONARY_KEY[self.scenario]
        lines = [
            "Observation Summary",
            f"Scenario: {self.scenario}",
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
            host_info = self._decode_host_row(row)
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

    def _decode_host_row(self, row: np.ndarray) -> dict:
        cfg = self.DICTIONARY_KEY[self.scenario]
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