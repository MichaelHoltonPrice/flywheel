"""Docker container launching for flywheel blocks.

Builds and runs Docker containers via subprocess, streaming output to
the terminal. Handles GPU access, shared memory, environment variables,
and volume mounts.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContainerConfig:
    """Configuration for a Docker container run.

    Attributes:
        image: Docker image name (with optional tag).
        gpus: Whether to pass --gpus all.
        shm_size: Shared memory size (e.g. "8g"), or None to use Docker default.
        env: Environment variables to set inside the container.
        mounts: List of (host_path, container_path, mode) tuples for -v mounts.
    """

    image: str
    gpus: bool = False
    shm_size: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ContainerResult:
    """Result of a container run.

    Attributes:
        exit_code: Process exit code (0 means success).
        elapsed_s: Wall-clock time in seconds.
    """

    exit_code: int
    elapsed_s: float


def build_docker_command(config: ContainerConfig, args: list[str] | None = None) -> list[str]:
    """Build the docker run command list from a ContainerConfig.

    Args:
        config: Container configuration specifying image, mounts, env, etc.
        args: Optional extra arguments to pass to the container entrypoint.

    Returns:
        A list of strings suitable for subprocess.Popen.
    """
    cmd = ["docker", "run", "--rm", "-t"]

    if config.gpus:
        cmd.extend(["--gpus", "all"])

    if config.shm_size is not None:
        cmd.extend(["--shm-size", config.shm_size])

    for key, value in config.env.items():
        cmd.extend(["-e", f"{key}={value}"])

    for host_path, container_path, mode in config.mounts:
        # Normalize Windows backslashes to forward slashes for Docker
        normalized_host = host_path.replace("\\", "/")
        cmd.extend(["-v", f"{normalized_host}:{container_path}:{mode}"])

    cmd.append(config.image)

    if args:
        cmd.extend(args)

    return cmd


def run_container(config: ContainerConfig, args: list[str] | None = None) -> ContainerResult:
    """Launch a Docker container and wait for it to complete.

    Builds a ``docker run`` command from the config, streams stdout and
    stderr to the terminal, and returns the exit code and elapsed time.

    Args:
        config: Container configuration specifying image, mounts, env, etc.
        args: Optional extra arguments to pass to the container entrypoint.

    Returns:
        A ContainerResult with exit code and wall-clock elapsed seconds.

    Raises:
        KeyboardInterrupt: Re-raised after terminating the container
            when the user presses Ctrl+C.
    """
    cmd = build_docker_command(config, args)

    # Prevent MSYS/Git Bash from translating Unix-style paths
    # (e.g., /output -> C:/Program Files/Git/output) in Docker commands.
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"

    start = time.monotonic()
    process = subprocess.Popen(cmd, env=env)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    elapsed = time.monotonic() - start

    return ContainerResult(exit_code=process.returncode, elapsed_s=elapsed)
