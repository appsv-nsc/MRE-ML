from trainers import AETrainer, RotTrainer,Mapping2DTrainer

import time
from utils.tools import get_logger, df_to_prettytable, save_table_as_image, write_header_to_csv, append_row_to_csv
import os
from utils.pipeline_tools import full_exp_monitor
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import pandas as pd

pretext_trainer_dict = {
    'autoencoder': AETrainer,
    'rot': RotTrainer
}


class full_pipeline_manager:
    def __init__(self, pipeline_cfg, pretext_cfg, downstream_cfg):
        self.pipeline_config = pipeline_cfg
        self.pretext_configs = pretext_cfg
        self.downstream_configs = downstream_cfg
        self.exps_monitor = []  # list of full_exp_monitors
        self.init_exp_logger()

    def init_exp_logger(self):
        self.run_id = '{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        setattr(self.pipeline_config, 'run_id', self.run_id)

        self.run_dir = '../../checkpoints' \
                       + '/{}/'.format(self.pipeline_config.run_mode) \
                       + '/{}/'.format(
            self.pipeline_config.run_id)  # path should be consistent with the one in utiles/recorder.py
        self.logger = get_logger(self.run_dir)
        self.save_tbx_log = self.run_dir + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

        self.run_monitor_detailed_csv_path = os.path.join(self.run_dir, 'run_monitor_results_detailed.csv')
        self.detailed_csv_headers = ['PipelineRunId', 'SubExpId',
                                     'PreTaskName', 'PreNetwork', 'PreTrainMetric', 'PreTrainResult',
                                     'PreValidMetric', 'PreValidResult', 'PreWeightsPath', 'PreLogsPath',
                                     'DownstrTaskName', 'DownstrNetwork', 'DownstrTrainMetric', 'DownstrTrainResult',
                                     'DownstrValidMetric', 'DownstrValidResult', 'DownstrWeightsPath',
                                     'DownstrLogsPath', 'DownstrTestResults', 'DownstrPathTestResults']

        write_header_to_csv(self.run_monitor_detailed_csv_path, self.detailed_csv_headers)

        self.run_monitor_abstract_csv_path = os.path.join(self.run_dir, 'run_monitor_results_abstract.csv')
        self.abstract_csv_headers = ['PipelineRunId', 'PreTaskName', 'PreNetwork',
                                     'DownstrNetwork', 'DownstrTrainResult', 'DownstrValidResult', 'DownstrTestResults']

        write_header_to_csv(self.run_monitor_abstract_csv_path, self.abstract_csv_headers)

    def prepare_common_downstream_steps(self):
        # set common task configs for training
        setattr(self.downstream_configs['task_cfgs'], 'resume', None)
        self.set_common_attributes(self.downstream_configs['task_cfgs'], mode='downstream')

    def set_common_attributes(self, config, mode='pretext'):
        if not hasattr(config, 'run_id'):
            setattr(config, 'run_id', self.pipeline_config.run_id)
        if not hasattr(config, 'gpu_ids'):
            setattr(config, 'gpu_ids', self.pipeline_config.gpu_ids)
        if not hasattr(config, 'benchmark'):
            setattr(config, 'benchmark', self.pipeline_config.benchmark)
        if not hasattr(config, 'manualseed'):
            setattr(config, 'manualseed', self.pipeline_config.manualseed)
        if not hasattr(config, 'run_mode'):
            setattr(config, 'run_mode', self.pipeline_config.run_mode)
        if not hasattr(config, 'save_tensorboard_graph'):
            setattr(config, 'save_tensorboard_graph', self.pipeline_config.save_tensorboard_graph)
        if not hasattr(config, 'dataset_name'):
            if mode == 'pretext':
                setattr(config, 'dataset_name', self.pipeline_config.pretext_dataset_name)
            else:
                setattr(config, 'dataset_name', self.pipeline_config.downstream_dataset_name)
        if not hasattr(config, 'dataset_path'):
            if mode == 'pretext':
                setattr(config, 'dataset_path', self.pipeline_config.pretext_preprocessed_data_path)
            else:
                setattr(config, 'dataset_path', self.pipeline_config.downstream_preprocessed_data_path)

    def run_downstream(self, network, current_exp_monitor, pretext_exists=True):
       
        if pretext_exists and current_exp_monitor.pretext_monitor.best_model_path is None:
            self.logger.info(
                "**********Unable to apply dwonstream training due to no training weights being saved for pretext task: " + current_exp_monitor.pretext_monitor.task_name + " using network: " + current_exp_monitor.pretext_monitor.network+"**********")
        else:
            # run downstream task training
            if pretext_exists:        
                setattr(self.downstream_configs['task_cfgs'], 'pretrained_model',
                        current_exp_monitor.pretext_monitor.best_model_path)
                setattr(self.downstream_configs['task_cfgs'], 'note',
                        current_exp_monitor.pretext_monitor.task_name)
            else:
                if not hasattr(self.downstream_configs['task_cfgs'], 'pretrained_model'):
                    setattr(self.downstream_configs['task_cfgs'], 'pretrained_model',
                            None) 
                setattr(self.downstream_configs['task_cfgs'], 'note',
                        "scratch")
                current_exp_monitor = full_exp_monitor(self.writer, len(self.exps_monitor) + 1)

            if self.pipeline_config.downstream_task in {'mapping'}:
                # set network skip//without
                if self.pipeline_config.use_skip_in_downstream:
                    current_network = network
                    setattr(self.downstream_configs['task_cfgs'], 'network', current_network)
                else:
                    current_network = network + '_WoSkip'
                    setattr(self.downstream_configs['task_cfgs'], 'network', current_network)

                if self.pipeline_config.downstream_task == 'mapping':
                    current_network = self.pipeline_config.downstream_task + '_' + network
                    DownstreamTrainer = Mapping2DTrainer(self.downstream_configs['task_cfgs'],
                                                     current_exp_monitor.downstream_monitor,
                                                     current_exp_monitor.pretext_monitor.task_name)

                current_exp_monitor.downstream_monitor.network = current_network
                current_exp_monitor.downstream_monitor.task_name = self.pipeline_config.downstream_task
            else:
                raise NotImplementedError

            DownstreamTrainer.train()


            if pretext_exists:
                self.logger.info(
                    "**********Finished downstream: " + self.pipeline_config.downstream_task + " training for pretext task: " + current_exp_monitor.pretext_monitor.task_name + " using network: " + current_network +"**********")
            
            else:
                self.logger.info(
                    "**********Finished downstream: " + self.pipeline_config.downstream_task + " training from scratch" + " using network: " + current_network +"**********")


    def run_pretext_downstream_exp(self, pretext_task_name):
        self.logger.info("**********Current pretext task: " + pretext_task_name + "**********")
        pretext_task_configs = self.pretext_configs[pretext_task_name]
        self.set_common_attributes(pretext_task_configs)

        for netwrk in self.pipeline_config.network_backbones:
            current_network = netwrk

            if pretext_task_name == 'autoencoder' and self.pipeline_config.downstream_task == 'mapping':
                current_network = pretext_task_name + '_' + netwrk
                if not self.pipeline_config.use_skip_in_pretext:
                    current_network = current_network + '_WoSkip'
            elif pretext_task_name == 'rot':
                current_network = pretext_task_name + '_' + netwrk + '_dense'

            setattr(pretext_task_configs, 'network', current_network)

            self.logger.info(
                "**********Current pretext task: " + pretext_task_name + ", current pretext network: " + current_network + "**********")

            current_exp_monitor = full_exp_monitor(self.writer, len(self.exps_monitor) + 1)
            current_exp_monitor.pretext_monitor.network = pretext_task_configs.network
            current_exp_monitor.pretext_monitor.task_name = pretext_task_name

            PretextTrainer = pretext_trainer_dict[pretext_task_name](pretext_task_configs,
                                                                     pipeline_task_monitor=current_exp_monitor.pretext_monitor)

            PretextTrainer.train()

            self.logger.info(
                "**********Finished: " + pretext_task_name + " training using network: " + current_network + ",  currently running downstream task**********")
            self.run_downstream(netwrk, current_exp_monitor)
            self.save_exp_monitor_data(current_exp_monitor)
            self.exps_monitor.append(current_exp_monitor)

    def save_exp_monitor_data(self, current_exp_monitor):
        new_row_detailed = {'PipelineRunId': self.pipeline_config.run_id, 'SubExpId': current_exp_monitor.exp_id,
                            'PreTaskName': current_exp_monitor.pretext_monitor.task_name,
                            'PreNetwork': current_exp_monitor.pretext_monitor.network,
                            'PreTrainMetric': current_exp_monitor.pretext_monitor.train_metric,
                            'PreTrainResult': current_exp_monitor.pretext_monitor.train_loss,
                            'PreValidMetric': current_exp_monitor.pretext_monitor.valid_metric,
                            'PreValidResult': current_exp_monitor.pretext_monitor.valid_result,
                            'PreWeightsPath': current_exp_monitor.pretext_monitor.best_model_path,
                            'PreLogsPath': current_exp_monitor.pretext_monitor.log_dir,
                            'DownstrTaskName': current_exp_monitor.downstream_monitor.task_name,
                            'DownstrNetwork': current_exp_monitor.downstream_monitor.network,
                            'DownstrTrainMetric': current_exp_monitor.downstream_monitor.train_metric,
                            'DownstrTrainResult': current_exp_monitor.downstream_monitor.train_loss,
                            'DownstrValidMetric': current_exp_monitor.downstream_monitor.valid_metric,
                            'DownstrValidResult': current_exp_monitor.downstream_monitor.valid_result,
                            'DownstrWeightsPath': current_exp_monitor.downstream_monitor.best_model_path,
                            'DownstrLogsPath': current_exp_monitor.downstream_monitor.log_dir}

        # Append the new row to the Monitor DataFrame
        append_row_to_csv(self.run_monitor_detailed_csv_path, new_row_detailed, headers=self.detailed_csv_headers)
        new_row_detailed_as_str = '<br>'.join(f"{key}: {value}" for key, value in new_row_detailed.items())
        self.writer.add_text("ExperimentSummary/" + str(current_exp_monitor.exp_id), new_row_detailed_as_str,
                             global_step=0)

        new_row_abstract = {
            'PipelineRunId': self.pipeline_config.run_id,
            'PreTaskName': current_exp_monitor.pretext_monitor.task_name,
            'PreNetwork': current_exp_monitor.pretext_monitor.network,
            'DownstrNetwork': current_exp_monitor.downstream_monitor.network,
            'DownstrTrainResult': current_exp_monitor.downstream_monitor.train_loss,
            'DownstrValidResult': current_exp_monitor.downstream_monitor.valid_result}

        # Append the new row to the Monitor DataFrame
        append_row_to_csv(self.run_monitor_abstract_csv_path, new_row_abstract, headers=self.abstract_csv_headers)

    def run_pipeline(self):
        self.logger.info("**********Full pipeline experiment started**********")
        self.logger.info("**********Expirement Directory: " + self.run_dir + "**********")

        # apply downstream preprocessing if needed and prepare common downstream configs
        self.prepare_common_downstream_steps()

        for pretext_task in self.pipeline_config.pretext_tasks_to_apply:
            self.run_pretext_downstream_exp(pretext_task_name=pretext_task)
            self.logger.info(
                "**********Finished " + pretext_task + " pretext task with downstream evaluation for all network backbones**********")

        if len(self.pipeline_config.pretext_tasks_to_apply) == 0:
            for netwrk in self.pipeline_config.network_backbones:
                current_network = netwrk
                self.run_downstream(current_network, current_exp_monitor=None, pretext_exists=False)

        monitor_df_detailed = pd.read_csv(self.run_monitor_detailed_csv_path)
        monitor_table_detailed = df_to_prettytable(monitor_df_detailed)

        monitor_df_abstract = pd.read_csv(self.run_monitor_abstract_csv_path)
        monitor_df_abstract = df_to_prettytable(monitor_df_abstract)

        self.logger.info("**********Printing Detailed Experiments Summary**********")
        self.logger.info(monitor_table_detailed)
        self.logger.info("**********Printing Abstract Experiments Summary**********")
        self.logger.info(monitor_df_abstract)

        table_img = save_table_as_image(monitor_df_abstract, save_path=self.run_dir)
        self.writer.add_image('ExperimentsAbstractSummary', table_img, global_step=0)
        self.writer.add_text("ExperimentSummary/",
                             "******Detailed Experiments Summary Saved as Excel at:" + str(self.run_dir) + "******\n")

        self.writer.close()
