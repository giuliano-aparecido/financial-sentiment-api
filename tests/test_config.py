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


# --- model registry from env ---


def test_inference_urls_from_env_registers_every_suffixed_var_by_lowercased_prefix():
    env = {
        "LLAMA_INFERENCE_URL": "https://llama.modal.run",
        "APERTUS_INFERENCE_URL": "https://apertus.modal.run",
        "KIM_INFERENCE_URL": "https://kim.modal.run",
        "MY_KIM_INFERENCE_URL": "https://my-kim.modal.run",
    }
    assert config_module.inference_urls_from_env(env) == {
        "llama": "https://llama.modal.run",
        "apertus": "https://apertus.modal.run",
        "kim": "https://kim.modal.run",
        "my_kim": "https://my-kim.modal.run",
    }


def test_inference_urls_from_env_ignores_blank_values_and_unrelated_vars():
    env = {
        "KIM_INFERENCE_URL": "   ",
        "APERTUS_INFERENCE_URL_BACKUP": "https://nope.modal.run",
        "DATABASE_CONNECTION_STRING": "postgresql://x",
        "LLAMA_INFERENCE_URL": " https://llama.modal.run ",
    }
    assert config_module.inference_urls_from_env(env) == {"llama": "https://llama.modal.run"}


def test_inference_urls_from_env_rejects_a_var_whose_prefix_is_not_a_valid_model_name():
    with pytest.raises(ValueError, match="Invalid model name"):
        config_module.inference_urls_from_env({"_INFERENCE_URL": "https://x.modal.run"})


@pytest.mark.parametrize("raw, expected", [("llama", "llama"), (" Kim ", "kim"), ("qwen-2.5", "qwen-2.5"), ("my_model-v2", "my_model-v2"), ("../etc", None), ("", None), ("a" * 65, None)])
def test_normalize_model_name(raw, expected):
    if expected is None:
        with pytest.raises(ValueError, match="Invalid model name"):
            config_module.normalize_model_name(raw)
    else:
        assert config_module.normalize_model_name(raw) == expected

