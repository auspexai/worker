"""Tests for WorkerConfig.load() TOML parsing."""

from __future__ import annotations

from pathlib import Path

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
