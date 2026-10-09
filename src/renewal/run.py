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
from renewal.logging_setup import (
    attach_file_log,
    configure_logging,
    detach_file_log,
    run_log_scope,
    run_scope_filter,
)
from renewal.metrics.analyze import run_analyze
from renewal.metrics.base import build_metric_runner, enabled_metrics
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


def build_interventions(config: RunConfig, seed: int) -> list:
    """Build each configured intervention, passing the run seed to those that
    opt in with ``uses_run_seed`` and the looping model's token budget to those
    that opt in with ``uses_loop_max_tokens`` (so a scaffolding LLM can size its
    summaries to the model it steers)."""
    interventions: list = []
    for iv in config.interventions:
        target = Registry.get("intervention", iv.type)
        options = dict(iv.options)
        if getattr(target, "uses_run_seed", False):
            options["seed"] = seed
        if getattr(target, "uses_loop_max_tokens", False):
            options["loop_max_tokens"] = config.agents.params.max_tokens
        interventions.append(target(**options))
    return interventions


def _attach_llm_call_log(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "llm_calls.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(run_scope_filter(str(run_dir)))
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
    force_metrics: bool = False,
) -> dict:
    recorder = RunRecorder(run_dir)
    recorder.start(seed=seed, config=config)
    llm_call_handler = _attach_llm_call_log(run_dir) if debug else None
    try:
        agents = build_agents(config, seed)
        interventions = build_interventions(config, seed)
        scheduler = Scheduler(len(agents), seed)

        enabled = enabled_metrics(config, force=force_metrics)
        runner = build_metric_runner(
            config,
            run_id=recorder.run_id,
            seed=seed,
            experiment=config.logging.experiment_name,
            condition=config.logging.labels.get("condition"),
            run_dir=run_dir,
            enabled=enabled,
        )
        loop = Loop(
            agents,
            scheduler,
            interventions,
            recorder,
            window_size=config.metrics.window_size if runner else None,
            on_window=runner.on_window if runner else None,
            memory_turns=config.run.memory_turns,
        )
        if runner is not None:
            recorder.set_meta(metrics_status="running")
        await loop.run(config.run.rounds)
        if runner is not None:
            recorder.set_meta(metrics_status=await runner.finalize())
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
        f"parallel: {config.run.parallel or 'all'}  seed: {config.run.seed}"
    )
    memory = "unlimited" if config.run.memory_turns is None else config.run.memory_turns
    print(f"memory_turns: {memory}")
    print(f"agents (n={config.agents.n}):")
    for i, p in enumerate(config.agents.pool):
        print(f"  agent_{i}: {p.provider}/{p.model}")
    print(f"interventions: {[iv.type for iv in config.interventions] or 'none'}")
    print(f"metrics: {config.metrics.enabled or 'none'}")
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
    run_parser.add_argument("--metrics", action="store_true", help="Force metrics on (defaults to the Kong set when metrics.enabled is empty)")

    analyze_parser = subparsers.add_parser("analyze", help="Compute cross_run_sim over a run directory tree")
    analyze_parser.add_argument("experiment_dir", type=Path, help="Experiment directory of run dirs")
    analyze_parser.add_argument("--condition", default=None, help="Only analyze runs with this condition label")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "analyze":
        _cmd_analyze(args)


def _plan_replicates(
    config: RunConfig,
    out_dir: Path,
    base_seed: int,
    source_path: Path | None,
) -> list[tuple[int, int, Path]]:
    """Allocate one run directory per replicate and snapshot its inputs.

    Directories are allocated (and inputs snapshotted) up front, before any
    replicate starts, so the check-then-create in ``_unique_dir`` never runs
    concurrently. Replicate ``i`` uses seed ``base_seed + i``.
    """
    plans: list[tuple[int, int, Path]] = []
    for replicate in range(config.run.replicates):
        seed = base_seed + replicate
        run_dir = allocate_run_dir(
            out_dir,
            config.logging.experiment_name,
            seed,
            config.logging.labels,
        )
        snapshot_run_inputs(run_dir, config, source_path=source_path)
        plans.append((replicate, seed, run_dir))
    return plans


async def _run_one_replicate(
    config: RunConfig,
    *,
    replicate: int,
    seed: int,
    run_dir: Path,
    debug: bool,
    force_metrics: bool,
    semaphore: asyncio.Semaphore,
) -> Exception | None:
    """Run one replicate under its own scoped file log.

    Returns the exception if the replicate failed, else ``None``. A failure is
    reported but never cancels sibling replicates.
    """
    key = str(run_dir)
    async with semaphore:
        handler = attach_file_log(
            run_dir / "run.log",
            config.logging.file_level,
            run_key=key,
        )
        try:
            with run_log_scope(key):
                try:
                    summary = await run_replicate(
                        config,
                        run_dir=run_dir,
                        seed=seed,
                        debug=debug,
                        force_metrics=force_metrics,
                    )
                except Exception as error:  # noqa: BLE001 - report, keep siblings running
                    LOGGER.error("replicate %s failed: %s", replicate, error)
                    return error
                LOGGER.info("replicate %s: %s", replicate, summary)
                return None
        finally:
            detach_file_log(handler)


async def _run_replicates(
    config: RunConfig,
    plans: list[tuple[int, int, Path]],
    *,
    debug: bool,
    force_metrics: bool,
) -> list[Exception]:
    """Run every planned replicate concurrently, bounded by ``run.parallel``."""
    limit = config.run.parallel or len(plans) or 1
    semaphore = asyncio.Semaphore(limit)
    results = await asyncio.gather(
        *(
            _run_one_replicate(
                config,
                replicate=replicate,
                seed=seed,
                run_dir=run_dir,
                debug=debug,
                force_metrics=force_metrics,
                semaphore=semaphore,
            )
            for replicate, seed, run_dir in plans
        )
    )
    return [error for error in results if error is not None]


def _cmd_run(args) -> None:
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
    plans = _plan_replicates(config, out_dir, base_seed, args.config)
    concurrency = config.run.parallel or len(plans)
    LOGGER.info(
        "running %s replicate(s), max concurrency %s",
        len(plans),
        concurrency,
    )
    failures = asyncio.run(
        _run_replicates(
            config,
            plans,
            debug=args.debug,
            force_metrics=args.metrics,
        )
    )
    if failures:
        LOGGER.error("%s replicate(s) failed", len(failures))
        raise SystemExit(1)


def _cmd_analyze(args) -> None:
    configure_logging(console_level="INFO")
    out_path = run_analyze(args.experiment_dir, args.condition)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
