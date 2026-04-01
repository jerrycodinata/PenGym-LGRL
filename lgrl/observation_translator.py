from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from nasim.envs.host_vector import HostVector
from nasim.envs.observation import Observation
from nasim.envs.utils import AccessLevel


class ObservationTranslator:
    """Translate PenGym/NASim observations into LLM-friendly text.

    Supported inputs:
    - dict with key ``textual_observation`` (returned directly)
    - flat vector observation (1D)
    - tensor observation (2D: host rows + aux row)
    """

    def __init__(self, observation: Optional[Any] = None):
        # Keep a default observation for backward compatibility with previous API.
        self.observation = observation

    def translate(self, observation: Optional[Any] = None) -> str:
        obs = self.observation if observation is None else observation
        if obs is None:
            return "No observation available."

        if isinstance(obs, dict) and "textual_observation" in obs:
            return str(obs["textual_observation"])

        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim == 1:
            arr = self._reshape_flat_observation(arr)
        elif arr.ndim != 2:
            return f"Unsupported observation shape: {arr.shape}"

        host_rows, aux_row = self._decode_rows(arr)
        return self._format_text(host_rows, aux_row)

    def _reshape_flat_observation(self, flat_obs: np.ndarray) -> np.ndarray:
        state_size = HostVector.state_size
        if state_size is None or state_size <= 0:
            raise ValueError(
                "HostVector layout is uninitialized. Create/reset an environment first "
                "so NASim host indices are available."
            )

        total = flat_obs.size
        if total % state_size != 0:
            raise ValueError(
                f"Flat observation length {total} is not divisible by host state size {state_size}."
            )

        rows = total // state_size
        if rows < 2:
            raise ValueError("Observation must contain at least one host row and one aux row.")

        return flat_obs.reshape(rows, state_size)

    def _decode_rows(self, obs_2d: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if obs_2d.shape[0] < 2:
            raise ValueError("Observation tensor must include host rows and one aux row.")

        host_count = obs_2d.shape[0] - 1
        state_shape = (host_count, obs_2d.shape[1])
        readable_obs = Observation.from_numpy(obs_2d, state_shape)
        return readable_obs.get_readable()

    def _format_text(self, host_rows: list[dict[str, Any]], aux_row: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("Observation Summary")
        lines.append(f"Hosts visible in tensor: {len(host_rows)}")

        discovered_hosts = 0
        compromised_hosts = 0

        for host in host_rows:
            address = host.get("Address", ("?", "?"))
            discovered = bool(host.get("Discovered", False))
            reachable = bool(host.get("Reachable", False))
            compromised = bool(host.get("Compromised", False))
            access = self._access_to_label(host.get("Access", 0))

            if discovered:
                discovered_hosts += 1
            if compromised:
                compromised_hosts += 1

            active_os = self._active_features(host, HostVector.os_idx_map.keys())
            active_services = self._active_features(host, HostVector.service_idx_map.keys())
            active_processes = self._active_features(host, HostVector.process_idx_map.keys())

            lines.append(
                f"- Host {address}: discovered={discovered}, reachable={reachable}, "
                f"compromised={compromised}, access={access}"
            )
            lines.append(
                f"  os={active_os if active_os else 'none'}, "
                f"services={active_services if active_services else 'none'}, "
                f"processes={active_processes if active_processes else 'none'}"
            )

        lines.append(
            "Auxiliary Flags: "
            f"success={bool(aux_row.get('Success', False))}, "
            f"connection_error={bool(aux_row.get('Connection Error', False))}, "
            f"permission_error={bool(aux_row.get('Permission Error', False))}, "
            f"undefined_error={bool(aux_row.get('Undefined Error', False))}"
        )
        lines.append(
            f"Counts: discovered_hosts={discovered_hosts}, compromised_hosts={compromised_hosts}"
        )

        return "\n".join(lines)

    @staticmethod
    def _active_features(host_row: dict[str, Any], names: Iterable[str]) -> list[str]:
        return [name for name in names if bool(host_row.get(name, False))]

    @staticmethod
    def _access_to_label(access: Any) -> str:
        try:
            level = int(round(float(access)))
        except (TypeError, ValueError):
            return str(access)

        if level == int(AccessLevel.ROOT):
            return "ROOT"
        if level == int(AccessLevel.USER):
            return "USER"
        if level == int(AccessLevel.NONE):
            return "NONE"
        return str(level)
    