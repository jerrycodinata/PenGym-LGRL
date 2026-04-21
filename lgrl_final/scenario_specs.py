from typing import Dict


SCENARIO_SPECS: Dict[str, Dict[str, int]] = {
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
        "dims_y": 14,
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
        "dims_y": 14,
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
        "dims_y": 26,
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
        "dims_y": 32,
    },
}


def get_scenario_spec(scenario_name: str) -> Dict[str, int]:
    if scenario_name not in SCENARIO_SPECS:
        available = ", ".join(sorted(SCENARIO_SPECS.keys()))
        raise ValueError(f"Unsupported scenario '{scenario_name}'. Available: {available}")
    return SCENARIO_SPECS[scenario_name]
