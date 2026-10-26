import pathlib

import slm_eval


def test_package_imports():
    assert slm_eval.__version__ == "0.1.0"


def test_specs_present():
    root = pathlib.Path(__file__).resolve().parent.parent
    for doc in ("SPEC_V3.md", "SPEC_PRODUCTION.md"):
        assert (root / doc).exists(), f"missing {doc}"
