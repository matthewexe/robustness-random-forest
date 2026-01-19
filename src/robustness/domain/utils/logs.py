def init_dir(path: str) -> None:
    import os
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

def init_log_dirs() -> None:
    init_dir("logs")
    init_dir("logs/bdds")
    init_dir("logs/psf")
    init_dir("logs/tableau")
    init_dir("logs/robustness_dags")