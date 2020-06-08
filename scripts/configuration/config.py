import datetime as dt
import json
import logging
import multiprocessing
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel

# Defaults
# Refresh days
# Columns that are used to join stuff onto


class Configuration(BaseModel):
    name: str
    apikey: str
    offset: int
    tickers: str
    refresh_rate: int
    save_output: bool
    min_records: int
    min_volume: float
    min_price: float


def load_configuration() -> Configuration:
    name = os.getenv("NAMED_RUN")

    logging.info(f"Running {name}")

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scenarios.yml")
    with open(path) as f:
        scenarios = yaml.full_load(f)

    envs = {
        "apikey": os.getenv("API_KEY"),
        "refresh_rate": os.getenv("REFRESH_RATE_OVERIDE"),
        "save_output": os.getenv("SAVE_OUTPUT_OVERRIDE")
    }
    envs = {k: v for k, v in envs.items() if v is not None}

    # May add more in the future for default values for example
    defaults = scenarios["defaults"]
    name_args = scenarios["named_runs"][name] or {}
    kwargs = {**name_args, **defaults, **envs}
    kwargs["name"]=name

    config = Configuration(**kwargs)
    return config
