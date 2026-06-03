from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hardware_search_common import add_common_args, parse_int_list, rebuild_global_outputs, run_full_scenario


def _worker(args, sid: int) -> str:
    run_full_scenario(args, sid)
    return f's{sid}: full hardware-search run tamamlandı'


def parse_args():
    parser = argparse.ArgumentParser(description='full MPC hardware search: u1-f1-u2-f2-u3-f3-u4-final-search in one command')
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    scenario_ids = parse_int_list(args.scenario_ids) or [0]
    workers = max(1, min(int(args.num_workers), len(scenario_ids)))
    if workers <= 1:
        for sid in scenario_ids:
            print(_worker(args, int(sid)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, args, int(sid)) for sid in scenario_ids]
            for fut in as_completed(futures):
                print(fut.result())
    rebuild_global_outputs(output_dir)
    print('\nfull hardware-search run tamamlandı.')
    print(f'output_dir: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
