"""Settings: which key and endpoint the Spark client uses."""

import pytest

from lens import config
from lens.config import Settings

KEYS = ("SPARK_API_KEY", "BASETEN", "SPARK_BASE_URL", "SPARK_MODEL", "SPARK_PRICE_IN_PER_M", "SPARK_PRICE_OUT_PER_M")


@pytest.fixture
def env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)  # ignore the developer's .env
    return monkeypatch


def test_baseten_key_used_by_default(env):
    env.setenv("BASETEN", "b")
    s = Settings.from_env()
    assert s.baseten_key == "b"
    assert s.spark_base_url == "https://inference.baseten.co/v1"
    assert s.spark_model == "meta-models/muse-spark-open-1026"


def test_spark_api_key_wins_over_baseten(env):
    env.setenv("BASETEN", "b")
    env.setenv("SPARK_API_KEY", "g")
    env.setenv("SPARK_BASE_URL", "https://x--muna-lens-glimmer-serve.modal.run/v1")
    env.setenv("SPARK_MODEL", "muse-glimmer")
    env.setenv("SPARK_PRICE_IN_PER_M", "0")
    env.setenv("SPARK_PRICE_OUT_PER_M", "0")
    s = Settings.from_env()
    assert (s.baseten_key, s.spark_model) == ("g", "muse-glimmer")
    assert s.spark_base_url.endswith(".modal.run/v1")
    assert (s.spark_price_in, s.spark_price_out) == (0.0, 0.0)


def test_spark_api_key_alone_is_enough(env):
    env.setenv("SPARK_API_KEY", "g")
    assert Settings.from_env().baseten_key == "g"


def test_empty_spark_api_key_falls_back(env):
    env.setenv("SPARK_API_KEY", "")
    env.setenv("BASETEN", "b")
    assert Settings.from_env().baseten_key == "b"


def test_no_key_at_all(env):
    assert Settings.from_env().baseten_key is None
