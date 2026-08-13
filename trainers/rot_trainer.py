from trainers.base_trainer import BaseTrainer
import torch
import sys
from tqdm import tqdm
import numpy as np
import os


class RotTrainer(BaseTrainer):
    def __init__(self, config, pipeline_task_monitor=None):
        super(RotTrainer, self).__init__(config, pipeline_task_monitor)
        self.num_rotations_per_patch = 1

    def set_input(self, sample):
        input, target = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)

    def train(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        if self.part_of_pipeline:
            self.pipeline_monitor.set_metric_attributes(train_metric=self.config.loss + " loss", valid_metric="Acc.")

        if self.config.resume is not None:
            best_acc = self.config.resume_acc
        else:
            best_acc = 0
        best_model_path = None
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):

                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(round(self.loss.item(), 2))

                if (epoch == 1) and (itr == 1) and self.config.save_tensorboard_graph and (
                        'rpl' not in self.config.network):
                    self.recorder.writer.add_graph(self.model, self.input)
                elif (epoch == 1) and (itr == 1) and self.config.save_tensorboard_graph and (
                        'rpl' in self.config.network):
                    self.recorder.writer.add_graph(self.model, (self.uniform_patch, self.input))

                # print(self.input.size(), self.target.size(), self.pred.size())
                if (itr + 1) % 500 == 0:
                    self.recorder.logger.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                                              .format(epoch + 1, self.config.epochs, itr + 1, np.average(train_losses)))
                    sys.stdout.flush()

            # if epoch % self.config.val_epoch == 0:
            with torch.no_grad():
                # ACC
                valid_acc = 0
                total = 0
                self.model.eval()
                self.recorder.logger.info("validating....")
                for itr, sample in enumerate(self.eval_dataloader):
                    self.set_input(sample)
                    self.forward()
                    v_loss = self.criterion(self.pred, self.target)
                    valid_losses.append(v_loss.item())
                    pred = torch.softmax(self.pred.data, 1)
                    _, predicted_label = torch.max(pred, 1)
                    count = (predicted_label == self.target).sum()
                    valid_acc += count
                    total += self.target.size(0)
                valid_acc = valid_acc / total

            # logging
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            self.recorder.logger.info(
                "Epoch {}, validation loss is {:.4f}, training loss is {:.4f}, validation acc is {:.4f}".format(
                    epoch + 1, valid_loss,
                    train_loss, valid_acc))

            if self.part_of_pipeline:
                self.pipeline_monitor.pipeline_writer.add_scalar(
                    "Pretext/" + self.pipeline_monitor.task_name + "/" + str(self.config.network) + "/Loss/train",
                    train_loss, epoch)
                self.pipeline_monitor.pipeline_writer.add_scalar(
                    "Pretext/" + self.pipeline_monitor.task_name + "/" + str(self.config.network) + "/Acc/valid",
                    valid_loss, epoch)

            # reset
            train_losses = []
            valid_losses = []

            if valid_acc > best_acc:
                self.recorder.logger.info(
                    "Validation metric increases from {:.4f} to {:.4f}".format(best_acc, valid_acc))
                best_acc = valid_acc
                num_epoch_no_improvement = 0
                # save model
                best_model_path = os.path.join(self.recorder.save_dir, "SSM_ROT.pth")
                self.save_state_dict(epoch + 1, best_model_path)
                self.recorder.logger.info("Saving model{} ".format(os.path.join(self.recorder.save_dir, "SSM_ROT.pth")))

                if self.part_of_pipeline:
                    self.pipeline_monitor.set_best_model_trackers(train_loss=train_loss, valid_perf=valid_loss,
                                                                  stop_epoch=epoch + 1, best_model_path=best_model_path)
            else:
                self.recorder.logger.info(
                    "Validation metric does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_acc,
                                                                                                          num_epoch_no_improvement))
                num_epoch_no_improvement += 1

            if num_epoch_no_improvement == self.config.patience:
                self.recorder.logger.info("Early Stopping")
                break
            if self.scheduler is not None:
                if self.config.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()

            sys.stdout.flush()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return
