"""Docker container launching for flywheel blocks.

Builds and runs Docker containers via subprocess, streaming output to
the terminal. Handles environment variables, volume mounts, and
pass-through Docker flags.
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
        docker_args: Extra flags passed to ``docker run`` before the image
            name (e.g. ``["--gpus", "all", "--shm-size", "8g"]``).
            Project-specific; flywheel does not interpret these.
        env: Environment variables to set inside the container.
        mounts: List of (host_path, container_path, mode) tuples for -v mounts.
    """

    image: str
    docker_args: list[str] = field(default_factory=list)
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


def build_docker_command(
    config: ContainerConfig,
    args: list[str] | None = None,
    name: str | None = None,
) -> list[str]:
    """Build the docker run command list from a ContainerConfig.

    Args:
        config: Container configuration specifying image, mounts, env, etc.
        args: Optional extra arguments to pass to the container entrypoint.
        name: Optional container name (for ``docker kill`` by name).

    Returns:
        A list of strings suitable for subprocess.Popen.
    """
    cmd = ["docker", "run", "--rm", "-t"]

    if name:
        cmd.extend(["--name", name])

    cmd.extend(config.docker_args)

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


def start_container(
    config: ContainerConfig,
    args: list[str] | None = None,
    name: str | None = None,
    *,
    capture_output: bool = False,
) -> subprocess.Popen:
    """Launch a Docker container non-blockingly.

    Returns the live ``subprocess.Popen`` so callers can observe
    and control the container lifecycle (poll for exit, wait with
    timeout, signal by container name through ``docker kill``).
    Does not block on ``wait()`` — that's the caller's
    responsibility.

    Args:
        config: Container configuration specifying image, mounts,
            env, etc.
        args: Optional extra arguments for the container
            entrypoint.
        name: Optional container name.  Required if the caller
            intends to signal the container via ``docker kill``
            (which addresses containers by name).
        capture_output: When ``True``, wire the child process's
            stdout and stderr to pipes the caller can drain.
            Default ``False`` lets the streams inherit the parent's
            file descriptors (prints to the terminal).

    Returns:
        A running :class:`subprocess.Popen` for the container.
    """
    cmd = build_docker_command(config, args, name=name)
    # Prevent MSYS/Git Bash from translating Unix-style paths
    # (e.g., /output -> C:/Program Files/Git/output).
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    popen_kwargs: dict = {"env": env}
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
        popen_kwargs["text"] = True
        popen_kwargs["encoding"] = "utf-8"
    return subprocess.Popen(cmd, **popen_kwargs)


def run_container(
    config: ContainerConfig,
    args: list[str] | None = None,
    name: str | None = None,
) -> ContainerResult:
    """Launch a Docker container and wait for it to complete.

    Builds a ``docker run`` command from the config, streams stdout and
    stderr to the terminal, and returns the exit code and elapsed time.

    Args:
        config: Container configuration specifying image, mounts, env, etc.
        args: Optional extra arguments to pass to the container entrypoint.
        name: Optional container name (for ``docker kill`` by name).

    Returns:
        A ContainerResult with exit code and wall-clock elapsed seconds.

    Raises:
        KeyboardInterrupt: Re-raised after terminating the container
            when the user presses Ctrl+C.
    """
    start = time.monotonic()
    process = start_container(config, args, name=name)
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
