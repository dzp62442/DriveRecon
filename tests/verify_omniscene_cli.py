"""Exercise the real CLI's train/final-mini/resume/total paths on synthetic assets."""

import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.omniscene_helpers import fixture_assets
from train_omniscene import main


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = root/'synthetic_assets'
    fixture_assets(data)
    common = ['--config', 'configs/omniscene/112x200.py', '--work-dir', str(root/'run'), '--data-root', str(data),
              '--cfg-options', 'training.max_steps=1', 'training.validate_every_steps=1',
              'training.mini_every_n_validations=1', 'training.checkpoint_every_steps=1',
              'evaluation.time_skip_bins=0', 'feishu.enabled=False']
    main(common+['--mode', 'train'])
    log = root/'run/metrics.jsonl'
    records = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = [row['kind'] for row in records]
    assert kinds == ['train', 'validation', 'mini_periodic', 'mini_final'], kinds
    main(common+['--mode', 'train'])
    assert [json.loads(line) for line in log.read_text().splitlines()] == records
    main(common+['--mode', 'test', '--test-split', 'total', '--checkpoint', 'final'])
    path = root/'run/evaluation/total/step-00000001/evaluation_summary.json'
    summary = json.loads(path.read_text())
    assert summary['complete'] and summary['processed_bins'] == 2
    assert summary['groups']['novel_12']['num_bins'] == 2
    print('CLI train / periodic mini / final mini / completed-run resume / total: PASSED', flush=True)


if __name__ == '__main__':
    run()
