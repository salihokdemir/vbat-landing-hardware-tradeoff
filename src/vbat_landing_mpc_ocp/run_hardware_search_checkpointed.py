from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hardware_search_common import (
    add_common_args,
    parse_int_list,
    print_status,
    rebuild_global_outputs,
    run_feasible_pass,
    run_final_search_for_scenario,
    run_full_scenario,
    run_utopia_pass,
)


def _run_stage_for_scenario(args, sid: int) -> str:
    if args.stage == 'u1':
        run_utopia_pass(args, sid, pass_index=1)
    elif args.stage == 'f1':
        run_feasible_pass(args, sid, pass_index=1)
    elif args.stage == 'u2':
        run_utopia_pass(args, sid, pass_index=2)
    elif args.stage == 'f2':
        run_feasible_pass(args, sid, pass_index=2)
    elif args.stage == 'u3':
        run_utopia_pass(args, sid, pass_index=3)
    elif args.stage == 'f3':
        run_feasible_pass(args, sid, pass_index=3)
    elif args.stage == 'u4':
        run_utopia_pass(args, sid, pass_index=4)
    elif args.stage == 'final-search':
        run_final_search_for_scenario(args, sid)
    elif args.stage == 'all':
        run_full_scenario(args, sid)
    else:
        raise ValueError(args.stage)
    return f's{sid}: {args.stage} tamamlandı'


def parse_args():
    parser = argparse.ArgumentParser(description='checkpointed MPC hardware search: recursive contraction + final dual-direction subsets')
    parser.add_argument('--stage', choices=['u1', 'f1', 'u2', 'f2', 'u3', 'f3', 'u4', 'final-search', 'all', 'rebuild', 'status'], required=True)
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    scenario_ids = parse_int_list(args.scenario_ids) or [0]

    if args.stage == 'rebuild':
        rebuild_global_outputs(output_dir)
        print(f'Global CSVler yenilendi: {output_dir.resolve()}')
        return
    if args.stage == 'status':
        print_status(args)
        return

    workers = max(1, min(int(args.num_workers), len(scenario_ids)))

    def run_stage_for_all(stage_name: str) -> None:
        original_stage = args.stage
        args.stage = stage_name
        print(f'\n=== hardware-search stage başladı: {stage_name} ===')
        if workers <= 1:
            for sid in scenario_ids:
                print(_run_stage_for_scenario(args, int(sid)))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_run_stage_for_scenario, args, int(sid)) for sid in scenario_ids]
                for fut in as_completed(futures):
                    print(fut.result())
        rebuild_global_outputs(output_dir)
        args.stage = original_stage

    if args.stage == 'all':
        # Barriered all-run: u1 for every scenario, then f1 for every scenario,
        # etc. This prevents one fast scenario from entering final-search while
        # another scenario has not reached u4 yet.
        for stage_name in ('u1', 'f1', 'u2', 'f2', 'u3', 'f3', 'u4', 'final-search'):
            run_stage_for_all(stage_name)
    else:
        run_stage_for_all(args.stage)

    rebuild_global_outputs(output_dir)
    print('\ncheckpointed hardware-search stage tamamlandı.')
    print(f'output_dir: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
