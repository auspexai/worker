"""Tests for WorkerConfig.load() TOML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_worker.config import WorkerConfig


def _write(cfg_path: Path, body: str) -> None:
    cfg_path.write_text(body)


class TestConfigDefaults:
    def test_defaults_when_no_files(self, tmp_path: Path) -> None:
        cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
        # v0.1.2 flipped this default to the public coord URL. The
        # original lab default `http://127.0.0.1:8080` still works as
        # an override via AUSPEXAI_COORDINATOR_URL env or `[coordinator]
        # url` in worker.toml.
        assert cfg.coordinator_url == "https://coord.auspexai.network"
        assert cfg.heartbeat_interval_seconds == 60
        assert cfg.max_ram_gb is None
        assert cfg.declared_gpus.is_empty() is True


class TestSandboxPolicy:
    def test_defaults_permissive(self, tmp_path: Path) -> None:
        cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
        assert cfg.sandbox_policy == "permissive"
        assert cfg.sandbox_use_bubblewrap is True

    def test_strict_from_sandbox_block(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, "[sandbox]\npolicy = 'strict'\n")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.sandbox_policy == "strict"

    def test_invalid_policy_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, "[sandbox]\npolicy = 'wide-open'\n")
        with pytest.raises(ValueError, match="policy must be one of"):
            WorkerConfig.load(config_path=cfg_file, env={})


class TestExecutorAutoAcquire:
    def test_defaults_off(self, tmp_path: Path) -> None:
        cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
        assert cfg.auto_acquire is False

    def test_parsed_from_executor_block(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, "[executor]\nmode = 'provisioned'\nauto_acquire = true\n")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.auto_acquire is True
        assert cfg.execute_tenant_code == "provisioned"

    def test_env_override(self, tmp_path: Path) -> None:
        cfg = WorkerConfig.load(
            config_path=tmp_path / "missing.toml",
            env={"AUSPEXAI_WORKER_AUTO_ACQUIRE": "1"},
        )
        assert cfg.auto_acquire is True


class TestResourcesBlock:
    def test_parses_caps(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(
            cfg_file,
            "[resources]\nmax_ram_gb = 16\nmax_cpu_cores = 6\nnetwork_quota_mb_per_hour = 1000\n",
        )
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.max_ram_gb == 16.0
        assert cfg.max_cpu_cores == 6
        assert cfg.network_quota_mb_per_hour == 1000


class TestCapabilitiesGpusBlock:
    def test_full_declaration(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(
            cfg_file,
            "[capabilities.gpus]\n"
            "nvidia = 1\n"
            'nvidia_model = "RTX 4090"\n'
            "vram_total_gb = 24.0\n"
            "amd = false\n",
        )
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.declared_gpus.nvidia == 1
        assert cfg.declared_gpus.nvidia_model == "RTX 4090"
        assert cfg.declared_gpus.vram_total_gb == 24.0
        assert cfg.declared_gpus.amd is False

    def test_partial_declaration(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, "[capabilities.gpus]\nnvidia = 2\n")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.declared_gpus.nvidia == 2
        assert cfg.declared_gpus.nvidia_model is None
        assert cfg.declared_gpus.vram_total_gb is None
        assert cfg.declared_gpus.is_empty() is False

    def test_missing_block_leaves_declared_empty(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, '[coordinator]\nurl = "http://localhost:8080"\n')
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.declared_gpus.is_empty() is True


class TestEnvOverrides:
    def test_coordinator_url_env_wins(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "worker.toml"
        _write(cfg_file, '[coordinator]\nurl = "http://from-toml:8080"\n')
        cfg = WorkerConfig.load(
            config_path=cfg_file,
            env={"AUSPEXAI_COORDINATOR_URL": "http://from-env:18080"},
        )
        assert cfg.coordinator_url == "http://from-env:18080"


class TestInferenceConfig:
    """W-S (§9 #43) [inference] block."""

    def test_defaults_dormant(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text("", encoding="utf-8")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.inference_backend == "none"
        assert cfg.inference_ollama_url == "http://127.0.0.1:11434"

    def test_toml_block_parsed(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text(
            '[inference]\nbackend = "ollama"\nollama_url = "http://127.0.0.1:9999/"\n',
            encoding="utf-8",
        )
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.inference_backend == "ollama"
        assert cfg.inference_ollama_url == "http://127.0.0.1:9999"  # trailing / stripped

    def test_unknown_backend_rejected(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text('[inference]\nbackend = "vllm"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="backend must be one of"):
            WorkerConfig.load(config_path=cfg_file, env={})


class TestWorkerFlavor:
    """§9 #46 [worker] flavor — install-profile bookkeeping."""

    def test_defaults_none(self, tmp_path):
        cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
        assert cfg.flavor is None

    def test_toml_block_parsed(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text('[worker]\nflavor = "inference"\n', encoding="utf-8")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.flavor == "inference"

    def test_future_flavor_name_tolerated(self, tmp_path):
        # Shape-validated, NOT an enum: an old binary must tolerate a flavor
        # minted after it shipped.
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text('[worker]\nflavor = "tensor-2"\n', encoding="utf-8")
        assert WorkerConfig.load(config_path=cfg_file, env={}).flavor == "tensor-2"

    def test_bad_shape_rejected(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text('[worker]\nflavor = "Not A Flavor!"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="flavor must match"):
            WorkerConfig.load(config_path=cfg_file, env={})

    def test_env_override(self, tmp_path):
        cfg = WorkerConfig.load(
            config_path=tmp_path / "missing.toml",
            env={"AUSPEXAI_WORKER_FLAVOR": "full"},
        )
        assert cfg.flavor == "full"


class TestFlavorAndInferenceSetters:
    """set_worker_flavor / set_inference_backend — the onramp's config writes."""

    def test_set_worker_flavor_preserves_file(self, tmp_path):
        from auspexai_worker.config import set_worker_flavor

        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text(
            "# volunteer's comment\n[coordinator]\nurl = 'http://x:1'\n", encoding="utf-8"
        )
        assert set_worker_flavor(cfg_file, "inference") == "inference"
        text = cfg_file.read_text(encoding="utf-8")
        assert "# volunteer's comment" in text
        assert 'flavor = "inference"' in text
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.flavor == "inference"
        assert cfg.coordinator_url == "http://x:1"

    def test_set_worker_flavor_rejects_bad_shape(self, tmp_path):
        from auspexai_worker.config import set_worker_flavor

        with pytest.raises(ValueError):
            set_worker_flavor(tmp_path / "worker.toml", "BAD NAME")

    def test_set_inference_backend_round_trips(self, tmp_path):
        from auspexai_worker.config import set_inference_backend

        cfg_file = tmp_path / "worker.toml"
        assert set_inference_backend(cfg_file, "ollama") == "ollama"
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.inference_backend == "ollama"
        # second write replaces in place (no duplicate keys)
        set_inference_backend(cfg_file, "none")
        assert WorkerConfig.load(config_path=cfg_file, env={}).inference_backend == "none"
        assert cfg_file.read_text(encoding="utf-8").count("backend =") == 1

    def test_set_sandbox_policy_round_trips_and_preserves_other_sections(self, tmp_path):
        from auspexai_worker.config import set_sandbox_policy

        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text("[executor]\nmode = 'provisioned'\n")
        assert set_sandbox_policy(cfg_file, "strict") == "strict"
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.sandbox_policy == "strict"
        assert cfg.execute_tenant_code == "provisioned"  # other section untouched
        # second write replaces in place (no duplicate keys)
        set_sandbox_policy(cfg_file, "permissive")
        assert WorkerConfig.load(config_path=cfg_file, env={}).sandbox_policy == "permissive"
        assert cfg_file.read_text(encoding="utf-8").count("policy =") == 1

    def test_set_sandbox_policy_rejects_invalid(self, tmp_path):
        from auspexai_worker.config import set_sandbox_policy

        with pytest.raises(ValueError, match="policy must be one of"):
            set_sandbox_policy(tmp_path / "worker.toml", "wide-open")

    def test_set_auto_acquire_round_trips_and_preserves_execute_policy(self, tmp_path):
        from auspexai_worker.config import set_auto_acquire

        cfg_file = tmp_path / "worker.toml"
        # Non-default execution policy → preservation is observable.
        cfg_file.write_text("[executor]\nexecute_tenant_code = 'provisioned'\n")
        assert set_auto_acquire(cfg_file, True) is True
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.auto_acquire is True
        assert cfg.execute_tenant_code == "provisioned"  # policy untouched
        # second write replaces in place (no duplicate keys), flips the flag only
        set_auto_acquire(cfg_file, False)
        cfg2 = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg2.auto_acquire is False
        assert cfg2.execute_tenant_code == "provisioned"
        assert cfg_file.read_text(encoding="utf-8").count("auto_acquire =") == 1


class TestInferenceKeepAlive:
    def test_default_none(self, tmp_path):
        cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
        assert cfg.inference_keep_alive is None

    def test_parsed_from_toml(self, tmp_path):
        cfg_file = tmp_path / "worker.toml"
        cfg_file.write_text('[inference]\nkeep_alive = "0"\n', encoding="utf-8")
        cfg = WorkerConfig.load(config_path=cfg_file, env={})
        assert cfg.inference_keep_alive == "0"
