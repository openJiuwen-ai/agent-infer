# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Harbor Docker environment using images prepared before benchmark execution."""

import asyncio
import json

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.utils import default_docker_platform
from harbor.models.task.config import EnvironmentConfig
from harbor.utils.container_cache import docker_build_context_hash

from .terminalbench_images import ImageSpec, docker_platform, environment_image_specs, inspect_local_image


def sidecar_image_spec(target: str) -> ImageSpec:
    context = DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH
    digest = docker_build_context_hash(
        context=context, dockerfile_path=context / "Dockerfile", build_args={}, platform=target
    )
    return ImageSpec(f"{DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME}--{digest}", context)


class PreparedDockerEnvironment(DockerEnvironment):
    task_env_config: EnvironmentConfig

    async def start(self, force_build: bool):
        if force_build:
            raise ValueError("Terminal-Bench uses prepared images. Build them during image preparation.")
        if not self.task_env_config.docker_image:
            target = await asyncio.to_thread(docker_platform)
            specs = await asyncio.to_thread(environment_image_specs, self.environment_dir, None, target)
            image = specs[0].image
            self.task_env_config = self.task_env_config.model_copy(update={"docker_image": image})
            self._env_vars.prebuilt_image_name = image
        await super().start(force_build=force_build)

    @classmethod
    def _engine_cmd(cls, *args: str) -> list[str]:
        if args[:4] == ("container", "run", "--rm", cls._EGRESS_CONTROL_KERNEL_PROBE_IMAGE):
            args = (*args[:3], "--pull=never", *args[3:])
        return super()._engine_cmd(*args)

    async def _ensure_egress_control_sidecar_image_built(self) -> None:
        spec = sidecar_image_spec(await default_docker_platform())
        if await asyncio.to_thread(inspect_local_image, spec.image) is None:
            raise RuntimeError(f"Prepared Harbor egress image missing: {spec.image}")
        self._env_vars.egress_control_sidecar_image_name = spec.image

    async def _run_docker_compose_command(self, command: list[str], *args, **kwargs):
        if command[0] == "up":
            command = ["up", "--pull", "never", "--no-build", *command[1:]]
        if command[0] == "down" and "--rmi" in command:
            index = command.index("--rmi")
            command = command[:index] + command[index + 2 :]
        return await super()._run_docker_compose_command(command, *args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--describe-sidecar", required=True)
    spec = sidecar_image_spec(parser.parse_args().describe_sidecar)
    assert spec.context is not None
    print(
        json.dumps(
            {
                "image": spec.image,
                "context": str(spec.context),
                "probe_image": DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE,
            }
        )
    )
