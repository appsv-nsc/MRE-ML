import threading
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class mini_exp_monitor:
    """
    pretext only or downstream only
    """

    def __init__(self, pipeline_writer):
        self.task_name = None
        self.network = None
        self.log_dir = None
        self.train_metric = None
        self.train_loss = None
        self.valid_metric = None
        self.valid_result = None
        self.stop_epoch = None
        self.best_model_path = None
        self.pipeline_writer = pipeline_writer

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def set_metric_attributes(self, train_metric, valid_metric):
        self.train_metric = train_metric
        self.valid_metric = valid_metric

    def set_best_model_trackers(self, train_loss, valid_perf, stop_epoch, best_model_path):
        self.train_loss = train_loss
        self.valid_result = valid_perf
        self.stop_epoch = stop_epoch
        self.best_model_path = best_model_path


class full_exp_monitor:
    def __init__(self,pipline_writer, exp_id):
        self.pipline_writer = pipline_writer
        self.exp_id = exp_id
        self.pretext_monitor = mini_exp_monitor(self.pipline_writer)
        self.downstream_monitor = mini_exp_monitor(self.pipline_writer)
