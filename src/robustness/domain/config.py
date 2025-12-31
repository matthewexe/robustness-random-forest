from robustness.domain.utils.singleton import Singleton


class Config(Singleton):
    diagram_size: int
    prefix_var: str
