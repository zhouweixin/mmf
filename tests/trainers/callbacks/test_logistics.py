# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import unittest
from copy import deepcopy
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf

from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.models.base_model import BaseModel
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.utils.logger import Logger


class SimpleModule(BaseModel):
    def __init__(self, config={}):
        super().__init__(config)
        self.base = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5)
        )

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.classifier(self.base(x))
        return {"losses": {"total_loss": self.loss(x, target)}}


class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = list(range(1, 1001))

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class TestLogisticsCallback(unittest.TestCase):
    def setUp(self):
        self.trainer = argparse.Namespace()
        self.config = OmegaConf.create(
            {
                "model": "simple",
                "model_config": {},
                "training": {
                    "checkpoint_interval": 1,
                    "evaluation_interval": 10,
                    "early_stop": {"criteria": "val/total_loss"},
                    "batch_size": 16,
                    "log_interval": 10,
                    "logger_level": "info",
                },
            }
        )
        # Keep original copy for testing purposes
        self.trainer.config = deepcopy(self.config)
        self.trainer.writer = Logger(self.config)
        registry.register("writer", self.trainer.writer)
        self.report = Mock(spec=Report)
        self.report.dataset_name = "abcd"
        self.report.dataset_type = "test"

        self.trainer.model = SimpleModule()
        self.trainer.val_dataset = NumbersDataset()

        self.trainer.optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=1e-01
        )
        self.trainer.device = "cpu"
        self.trainer.num_updates = 0
        self.trainer.current_iteration = 0
        self.trainer.current_epoch = 0
        self.trainer.max_updates = 0
        self.trainer.meter = Meter()
        self.cb = LogisticsCallback(self.config, self.trainer)

    def test_on_train_start(self):
        self.cb.on_train_start()
        self.assertEqual(
            int(self.cb.train_timer.get_time_since_start().split("ms")[0]), 0,
        )

    def test_on_batch_end(self):
        self.cb.on_train_start()
        self.cb.on_batch_end(meter=self.trainer.meter, should_log=True)
        self.assertNotEqual(
            int(self.cb.train_timer.get_time_since_start().split("ms")[0]), 0,
        )
        self.assertNotEqual(
            int(self.cb.snapshot_timer.get_time_since_start().split("ms")[0]), 0,
        )

    def test_on_validation_start(self):
        self.cb.on_train_start()
        self.cb.on_validation_start()
        self.assertEqual(
            int(self.cb.snapshot_timer.get_time_since_start().split("ms")[0]), 0,
        )

    def test_on_validation_end(self):
        self.cb.on_train_start()
        self.assertEqual(
            int(self.cb.train_timer.get_time_since_start().split("ms")[0]), 0,
        )

    def test_on_test_end(self):
        self.cb.on_test_end(report=self.report, meter=self.trainer.meter)
        self.assertEqual(
            int(self.cb.total_timer.get_time_since_start().split("ms")[0]), 0,
        )
