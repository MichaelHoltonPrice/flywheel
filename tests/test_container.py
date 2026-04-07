from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flywheel.container import ContainerConfig, ContainerResult, build_docker_command, run_container


class TestBuildDockerCommand:
    def test_minimal_command(self):
        config = ContainerConfig(image="myimage:latest")
        cmd = build_docker_command(config)
        assert cmd == ["docker", "run", "--rm", "-t", "myimage:latest"]

    def test_with_gpus(self):
        config = ContainerConfig(image="train:latest", gpus=True)
        cmd = build_docker_command(config)
        assert "--gpus" in cmd
        idx = cmd.index("--gpus")
        assert cmd[idx + 1] == "all"

    def test_without_gpus(self):
        config = ContainerConfig(image="eval:latest", gpus=False)
        cmd = build_docker_command(config)
        assert "--gpus" not in cmd

    def test_with_shm_size(self):
        config = ContainerConfig(image="train:latest", shm_size="8g")
        cmd = build_docker_command(config)
        assert "--shm-size" in cmd
        idx = cmd.index("--shm-size")
        assert cmd[idx + 1] == "8g"

    def test_without_shm_size(self):
        config = ContainerConfig(image="train:latest")
        cmd = build_docker_command(config)
        assert "--shm-size" not in cmd

    def test_with_env_vars(self):
        config = ContainerConfig(
            image="train:latest",
            env={"CUDA_VISIBLE_DEVICES": "0", "BATCH_SIZE": "64"},
        )
        cmd = build_docker_command(config)
        assert "-e" in cmd
        assert "CUDA_VISIBLE_DEVICES=0" in cmd
        assert "BATCH_SIZE=64" in cmd

    def test_env_var_pairs(self):
        config = ContainerConfig(image="img:latest", env={"KEY": "VAL"})
        cmd = build_docker_command(config)
        idx = cmd.index("-e")
        assert cmd[idx + 1] == "KEY=VAL"

    def test_with_mounts(self):
        config = ContainerConfig(
            image="train:latest",
            mounts=[
                ("/host/data", "/container/data", "ro"),
                ("/host/output", "/container/output", "rw"),
            ],
        )
        cmd = build_docker_command(config)
        assert "-v" in cmd
        assert "/host/data:/container/data:ro" in cmd
        assert "/host/output:/container/output:rw" in cmd

    def test_windows_path_conversion(self):
        config = ContainerConfig(
            image="train:latest",
            mounts=[("C:\\Users\\test\\data", "/container/data", "ro")],
        )
        cmd = build_docker_command(config)
        # Backslashes should be converted to forward slashes
        mount_arg = [a for a in cmd if "/container/data" in a][0]
        assert "\\" not in mount_arg
        assert mount_arg == "C:/Users/test/data:/container/data:ro"

    def test_with_args(self):
        config = ContainerConfig(image="eval:latest")
        cmd = build_docker_command(config, args=["--epochs", "10"])
        assert cmd[-2:] == ["--epochs", "10"]

    def test_without_args(self):
        config = ContainerConfig(image="eval:latest")
        cmd = build_docker_command(config, args=None)
        assert cmd[-1] == "eval:latest"

    def test_empty_args_list(self):
        config = ContainerConfig(image="eval:latest")
        cmd = build_docker_command(config, args=[])
        assert cmd[-1] == "eval:latest"

    def test_full_config(self):
        config = ContainerConfig(
            image="train:latest",
            gpus=True,
            shm_size="16g",
            env={"LR": "0.001"},
            mounts=[("/data", "/input", "ro")],
        )
        cmd = build_docker_command(config, args=["--fast"])
        assert cmd[0:4] == ["docker", "run", "--rm", "-t"]
        assert "--gpus" in cmd
        assert "--shm-size" in cmd
        assert "-e" in cmd
        assert "-v" in cmd
        assert cmd[-1] == "--fast"

    def test_image_always_before_args(self):
        config = ContainerConfig(
            image="myimg:v2",
            gpus=True,
            env={"A": "B"},
            mounts=[("/x", "/y", "ro")],
        )
        cmd = build_docker_command(config, args=["--flag"])
        img_idx = cmd.index("myimg:v2")
        flag_idx = cmd.index("--flag")
        assert img_idx < flag_idx


class TestContainerResult:
    def test_fields(self):
        result = ContainerResult(exit_code=0, elapsed_s=12.5)
        assert result.exit_code == 0
        assert result.elapsed_s == 12.5

    def test_frozen(self):
        result = ContainerResult(exit_code=0, elapsed_s=1.0)
        with pytest.raises(AttributeError):
            result.exit_code = 1


class TestRunContainer:
    @patch("flywheel.container.subprocess.Popen")
    def test_calls_popen_with_correct_command(self, mock_popen):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = ContainerConfig(image="test:latest")
        run_container(config)

        mock_popen.assert_called_once_with(
            ["docker", "run", "--rm", "-t", "test:latest"]
        )
        mock_process.wait.assert_called_once()

    @patch("flywheel.container.subprocess.Popen")
    def test_returns_exit_code(self, mock_popen):
        mock_process = MagicMock()
        mock_process.returncode = 42
        mock_popen.return_value = mock_process

        config = ContainerConfig(image="test:latest")
        result = run_container(config)

        assert result.exit_code == 42

    @patch("flywheel.container.time.monotonic")
    @patch("flywheel.container.subprocess.Popen")
    def test_returns_elapsed_time(self, mock_popen, mock_time):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mock_time.side_effect = [100.0, 105.5]

        config = ContainerConfig(image="test:latest")
        result = run_container(config)

        assert result.elapsed_s == pytest.approx(5.5)

    @patch("flywheel.container.subprocess.Popen")
    def test_passes_args_to_command(self, mock_popen):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = ContainerConfig(image="test:latest")
        run_container(config, args=["--epochs", "5"])

        expected = ["docker", "run", "--rm", "-t", "test:latest", "--epochs", "5"]
        mock_popen.assert_called_once_with(expected)

    @patch("flywheel.container.subprocess.Popen")
    def test_gpu_flag_in_command(self, mock_popen):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        config = ContainerConfig(image="test:latest", gpus=True)
        run_container(config)

        called_cmd = mock_popen.call_args[0][0]
        assert "--gpus" in called_cmd


class TestContainerConfig:
    def test_frozen(self):
        config = ContainerConfig(image="img:latest")
        with pytest.raises(AttributeError):
            config.image = "other:latest"

    def test_defaults(self):
        config = ContainerConfig(image="img:latest")
        assert config.gpus is False
        assert config.shm_size is None
        assert config.env == {}
        assert config.mounts == []
