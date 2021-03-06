import logging
import time
from functools import wraps
from typing import Callable, List


def log_runtime(func: Callable) -> Callable:
    # From https://kedro.readthedocs.io/en/latest/03_tutorial/04_create_pipelines.html
    @wraps(func)
    def log_time(*args, **kwargs):
        log = logging.getLogger(__name__)
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start
        log.info("Running %r took %.2f seconds", func.__name__, elapsed)
        return result

    return log_time


def setup_logging(logger):

    # cannot use logging.basicConfig bc pystan calls basicConfig itself
    fmt = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # utc timestamps
    fmt.converter = time.gmtime

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(fmt)
    logger.addHandler(consoleHandler)
