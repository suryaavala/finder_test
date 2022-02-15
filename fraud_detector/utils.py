from pathlib import Path


def get_py_project_dir() -> Path:
    """Returns abspath of the current python project's base dir - so the dirpath of where setup.py resides

    Returns:
        Path: absolute path of this python project
    """
    return Path(__file__).resolve(strict=True).parents[1]


def get_abs_path(file_path: str) -> Path:
    """Takes in a relative (or absolute path) path, relative to the current python project's base directory (where setup.py resides), and returns the absolute path equivalent

    Args:
        file_path (str): relative path, relative to the current python project's base directory (where setup.py resides)

    Returns:
        PosixPath: returns the absolute path
    """  # noqa: E501

    return get_py_project_dir().joinpath(file_path).resolve(strict=True)
