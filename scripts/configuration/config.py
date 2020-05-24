import datetime as dt
import json
import logging
import multiprocessing
import os
from pydantic import BaseModel
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import yaml

# Defaults
# Refresh days
# Columns that are used to join stuff onto

class Configuration(BaseModel):
    apikey: str
    tickers: str
    refresh_rate: int

def load_configuration() -> Configuration:
    name = os.getenv("NAMED_RUN")

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scenarios.yml")
    with open(path) as f:
        scenarios = yaml.full_load(f)

    envs = {
        "apikey": os.getenv("API_KEY"),
        "refresh_rate": os.getenv("REFRESH_RATE_OVERIDE"),
    }
    envs = {k: v for k, v in envs.items() if v is not None}

    # May add more in the future for default values for example
    defaults = scenarios["defaults"]
    name_args = scenarios["named_runs"][name] or {}
    kwargs = {**name_args,**defaults,**envs}

    config =  Configuration(**kwargs)
    return config