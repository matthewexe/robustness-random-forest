from robustness.domain.logging import get_logger

logger = get_logger(__name__)


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            logger.debug(f"Creating new singleton instance for {cls.__name__}")
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            logger.debug(f"Returning existing singleton instance for {cls.__name__}")
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    pass