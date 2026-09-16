import pytest

import app.config as config_module


def test_parse_allowed_host_suffixes_lowercases_mixed_case_entries():
    # Regression: the suffix-parsing generator used to call only
    # suffix.strip(), not .lower() - is_allowed_inference_host compares
    # against a host that's already been lowercased by its caller (see
    # admin.py's update_inference_url), so a mixed-case
    # ALLOWED_INFERENCE_HOST_SUFFIXES override used to silently fail-closed
    # (every host rejected, since "mycompany.example.com" !=
    # "MyCompany.Example.COM") instead of matching as intended.
    assert config_module._parse_allowed_host_suffixes("MyCompany.Example.COM, Other.IO") == (
        "mycompany.example.com", "other.io",
    )


def test_is_allowed_inference_host_matches_after_a_mixed_case_override(monkeypatch):
    monkeypatch.setattr(
        config_module, "ALLOWED_INFERENCE_HOST_SUFFIXES",
        config_module._parse_allowed_host_suffixes("MyCompany.Example.COM"),
    )
    assert config_module.is_allowed_inference_host("mycompany.example.com") is True
    assert config_module.is_allowed_inference_host("sub.mycompany.example.com") is True
    assert config_module.is_allowed_inference_host("evil-mycompany.example.com") is False


def test_require_hf_api_token_raises_when_missing(monkeypatch):
    monkeypatch.setattr(config_module, "HF_API_TOKEN", None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        config_module.require_hf_api_token()


def test_require_hf_api_token_raises_when_blank(monkeypatch):
    monkeypatch.setattr(config_module, "HF_API_TOKEN", "")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        config_module.require_hf_api_token()


def test_require_hf_api_token_noop_when_set(monkeypatch):
    monkeypatch.setattr(config_module, "HF_API_TOKEN", "some-real-token")
    config_module.require_hf_api_token()  # must not raise
