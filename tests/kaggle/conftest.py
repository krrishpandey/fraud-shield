import importlib.util
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parents[2] / "kaggle"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, KAGGLE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def bundle_mod():
    return _load("build_kaggle_bundle")


@pytest.fixture(scope="session")
def import_mod():
    return _load("import_kaggle_output")
