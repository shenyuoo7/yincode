import importlib

import pytest
import yaml


def config_module():
    return importlib.import_module("yincode.config")


def valid_provider():
    return {"name": "local", "protocol": "openai", "api_key": "test-secret", "model": "test-model"}


def test_load_valid_config_and_hide_key_in_repr(tmp_path):
    mod = config_module()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"providers": [valid_provider()]}), encoding="utf-8")
    cfg = mod.load(path)
    assert cfg.providers[0].name == "local"
    assert cfg.providers[0].thinking is False
    assert cfg.providers[0].base_url is None
    assert "test-secret" not in repr(cfg)


@pytest.mark.parametrize(
    "data", [None, [], "value", {"providers": []}, {"providers": {}}, {"providers": [False]}]
)
def test_reject_wrong_config_structure(tmp_path, data):
    mod = config_module()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(mod.ConfigError):
        mod.load(path)


@pytest.mark.parametrize("field", ["name", "protocol", "api_key", "model"])
@pytest.mark.parametrize("value", [None, " ", 1, True, []])
def test_reject_invalid_required_field_without_echoing_values(tmp_path, field, value):
    mod = config_module()
    provider = valid_provider()
    provider[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"providers": [provider]}), encoding="utf-8")
    with pytest.raises(mod.ConfigError) as error:
        mod.load(path)
    assert field in str(error.value)
    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [("protocol", "invalid"), ("thinking", "true"), ("base_url", 123), ("base_url", " ")],
)
def test_reject_invalid_optional_fields(tmp_path, field, value):
    mod = config_module()
    provider = valid_provider()
    provider[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"providers": [provider]}), encoding="utf-8")
    with pytest.raises(mod.ConfigError, match=field):
        mod.load(path)


def test_missing_file_is_readable_error(tmp_path):
    mod = config_module()
    with pytest.raises(mod.ConfigError, match="不存在"):
        mod.load(tmp_path / "missing.yaml")


def test_malformed_yaml_does_not_echo_source_or_key(tmp_path):
    mod = config_module()
    path = tmp_path / "config.yaml"
    path.write_text("providers: [api_key: test-secret", encoding="utf-8")
    with pytest.raises(mod.ConfigError) as error:
        mod.load(path)
    assert "YAML" in str(error.value)
    assert "test-secret" not in str(error.value)


def test_invalid_utf8_is_config_error(tmp_path):
    mod = config_module()
    path = tmp_path / "config.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(mod.ConfigError):
        mod.load(path)
