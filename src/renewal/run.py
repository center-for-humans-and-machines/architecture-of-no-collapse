"""CLI entry point: ``renewal run <config> [options]``.

Loads a run config, allocates one self-contained run directory per replicate,
runs the loop, and writes artifacts. ``--dry-run`` validates and prints the
resolved plan without calling any model.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from renewal.artifacts import (
    RunRecorder,
    allocate_run_dir,
    snapshot_run_inputs,
)
from renewal.config import RunConfig, load_run_config
from renewal.core.agent import Agent
from renewal.core.loop import Loop
from renewal.core.scheduler import Scheduler
from renewal.logging_setup import attach_file_log, configure_logging, detach_file_log
from renewal.prompts.loader import PromptSource
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)
LLM_CALL_LOGGER = logging.getLogger("renewal.llm.calls")


def build_agents(config: RunConfig, seed: int) -> list[Agent]:
    pool = config.agents.pool
    prompt_source = PromptSource.load(
        config.agents.prompt_source,
        config.agents.system_prompt,
    )
    prompts = prompt_source.assign(
        config.agents.n,
        seed,
        overrides=config.agents.prompts_override,
    )
    params = config.agents.params.model_dump(exclude_none=True)
    agents: list[Agent] = []
    for i, llm_cfg in enumerate(pool):
        agents.append(
            Agent(
                name=f"agent_{i}",
                llm=llm_cfg.materialize(),
                system_prompt=prompts[i],
                params=params,
                meta={"provider": llm_cfg.provider, "model": llm_cfg.model},
            )
        )
    return agents


def build_interventions(config: RunConfig) -> list:
    return [Registry.get("intervention", iv.type)() for iv in config.interventions]


def _attach_llm_call_log(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "llm_calls.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LLM_CALL_LOGGER.addHandler(handler)
    LLM_CALL_LOGGER.setLevel(logging.INFO)
    LLM_CALL_LOGGER.propagate = False
    return handler


async def run_replicate(
    config: RunConfig,
    *,
    run_dir: Path,
    seed: int,
    debug: bool = False,
) -> dict:
    recorder = RunRecorder(run_dir)
    recorder.start(seed=seed, config=config)
    llm_call_handler = _attach_llm_call_log(run_dir) if debug else None
    try:
        agents = build_agents(config, seed)
        interventions = build_interventions(config)
        scheduler = Scheduler(len(agents), seed)
        loop = Loop(agents, scheduler, interventions, recorder)
        await loop.run(config.run.rounds)
    except BaseException as error:
        recorder.fail(error)
        raise
    finally:
        if llm_call_handler is not None:
            LLM_CALL_LOGGER.removeHandler(llm_call_handler)
            llm_call_handler.close()
    recorder.finish()
    LOGGER.info("run %s completed in %s", recorder.run_id, run_dir)
    return {
        "run_id": recorder.run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "turns": scheduler.turn_index + 1,
    }


def _print_dry_run(config: RunConfig) -> None:
    print(f"experiment: {config.logging.experiment_name}")
    print(
        f"rounds: {config.run.rounds}  replicates: {config.run.replicates}  "
        f"seed: {config.run.seed}"
    )
    print(f"agents (n={config.agents.n}):")
    for i, p in enumerate(config.agents.pool):
        print(f"  agent_{i}: {p.provider}/{p.model}")
    print(f"interventions: {[iv.type for iv in config.interventions] or 'none'}")
    print(f"out_dir: {config.logging.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="renewal", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one config file")
    run_parser.add_argument("config", type=Path, help="Run YAML file")
    run_parser.add_argument("--out", type=Path, default=None, help="Override out_dir")
    run_parser.add_argument("--seed", type=int, default=None, help="Override base seed")
    run_parser.add_argument("--debug", "-d", action="store_true", help="DEBUG logs + llm_calls.jsonl")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without running")
    args = parser.parse_args()

    try:
        config = load_run_config(args.config)
    except (ValueError, KeyError, yaml.YAMLError, OSError) as exc:
        print(f"error: invalid config {args.config}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.dry_run:
        _print_dry_run(config)
        return

    configure_logging(console_level=config.logging.console_level)
    out_dir = args.out or config.logging.out_dir
    base_seed = args.seed if args.seed is not None else config.run.seed

    LOGGER.info(
        "loaded config %s (rounds=%s, replicates=%s)",
        args.config,
        config.run.rounds,
        config.run.replicates,
    )
    for replicate in range(config.run.replicates):
        seed = base_seed + replicate
        run_dir = allocate_run_dir(
            out_dir,
            config.logging.experiment_name,
            seed,
            config.logging.labels,
        )
        snapshot_run_inputs(run_dir, config, source_path=args.config)
        file_handler = attach_file_log(run_dir / "run.log", config.logging.file_level)
        try:
            summary = asyncio.run(
                run_replicate(config, run_dir=run_dir, seed=seed, debug=args.debug)
            )
            LOGGER.info("replicate %s: %s", replicate, summary)
        finally:
            detach_file_log(file_handler)


if __name__ == "__main__":
    main()
