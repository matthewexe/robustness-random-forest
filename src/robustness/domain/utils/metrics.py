from functools import wraps

from robustness.domain.logging import get_logger
from time import perf_counter


def log_perf_counter(f):
    logger = get_logger(f.__module__)

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            logger.debug(
                "Function %s finished in %.6f seconds",
                wrapper.__name__,
                elapsed,
            )

    return wrapper