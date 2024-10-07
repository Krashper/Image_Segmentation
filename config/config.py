from environs import Env


def load_data_folder_path(path: str | None = None) -> str:
    env = Env()
    env.read_env(path)
    return env("DATA_FOLDER")