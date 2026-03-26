from __future__ import annotations

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable


def format_sample_id(sample_index: int, digits: int = 5) -> str:
    """Generate a sample ID string from a sample index"""
    return str(sample_index).zfill(digits)


class BatchContext:
    def __init__(
        self,
        max_workers: int,
        generator: Callable[..., list[tuple]],
        command: str,
        cli_flags: tuple[str],
        executable: str = "msc",
        global_flags: tuple[str] = ("--quiet", "--no-log"),
    ) -> None:
        self.max_workers: int = max_workers
        self.generator: Callable[..., list[tuple]] = generator
        self.command: str = command
        self.cli_flags: tuple[str] = cli_flags
        self.executable: str = executable
        self.global_flags: tuple[str] = global_flags

    def run_instance(self, sample_id: str, *args: Any) -> None:
        """
        Run a single instance of the batched process
        Args:
            sample_id (str): The unique identifier for the sample
            *args (Any): The arguments to be passed to the command, in the same order as cli_flags
        """
        assert len(args) == len(
            self.cli_flags
        ), "Number of arguments must match number of CLI flags"

        cmd = []
        cmd.append(self.executable)
        cmd.extend(self.global_flags)
        cmd.append(self.command)
        for flag, arg in zip(self.cli_flags, args, strict=True):
            cmd.append(flag)
            cmd.append(str(arg))
        cmd.append(sample_id)

        print(f"Running cmd: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        return

    def run_batch(self) -> int:
        """
        Generic method to run a batch of instances in parallel using ProcessPoolExecutor
        """
        workload = self.generator()
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.run_instance, *item) for item in workload]
            for future in as_completed(futures):
                future.result()

        return 0
