
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, TypeVar

import yaml


def get_digit_subscript(i: int):
    assert i in range(10), (
        f"Subscripts are only available for digits 0-9, but got {i}."
    )
    return chr(0x2080 + i)


def all_equal(values: List):
    return len(set(values)) <= 1


def load_config(path: Path) -> Dict:
    """
    Load a YAML configuration file, including scientific notation
    such as 5e-6.
    """
    loader = yaml.SafeLoader

    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            r"""^(?:
            [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+ \.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    path = Path(path)

    with path.open("r") as f:
        config = yaml.load(f, Loader=loader)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(config).__name__}"
        )

    return config


KeyType = TypeVar("KeyType")


def deep_update(
    mapping: Dict[KeyType, Any],
    *updating_mappings: Dict[KeyType, Any],
) -> Dict[KeyType, Any]:
    """
    Recursively update nested dictionaries without modifying
    the original mappings.
    """
    updated_mapping = deepcopy(mapping)

    for updating_mapping in updating_mappings:
        for key, value in updating_mapping.items():

            if (
                key in updated_mapping
                and isinstance(updated_mapping[key], dict)
                and isinstance(value, dict)
            ):
                updated_mapping[key] = deep_update(
                    updated_mapping[key],
                    value,
                )
            else:
                updated_mapping[key] = deepcopy(value)

    return updated_mapping
