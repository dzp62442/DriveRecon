"""Existing send_feishu integration; no credential handling or network dependency in training."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import logging
from pathlib import Path
import sys


class Notifier:
    def __init__(self, cfg):
        self.send = None
        self.executor = None
        if cfg.get('enabled', True):
            root = str(Path(cfg['module_root']).expanduser())
            if root not in sys.path:
                sys.path.append(root)
            try:
                self.send = importlib.import_module('auto_monitor.send_feishu').send_feishu
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='feishu')
            except ImportError as error:
                logging.warning('Feishu module unavailable: %s; training will continue', error)

    def __call__(self, title, body):
        if self.executor is not None:
            self.executor.submit(self._deliver, title, body)

    def _deliver(self, title, body):
        try:
            if not self.send(title, body):
                logging.warning('Feishu notification failed: %s', title)
        except Exception:
            logging.exception('Feishu notification failed; local results are already saved')

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
