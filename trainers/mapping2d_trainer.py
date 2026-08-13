"""
This trainer is designed for 2D Mapping
"""

from trainers.base_trainer import *
from utils.metrics import dice as cal_dice
from torch.functional import F


class Mapping2DTrainer(BaseTrainer):
    def __init__(self, config, pipeline_task_monitor=None, pipeline_pretext_name=None):
        super(Mapping2DTrainer, self).__init__(config, pipeline_task_monitor)
        self.pipeline_pretext_name = pipeline_pretext_name
        self.num_class = self.config.class_num
        self.stats_path = os.path.join(self.recorder.save_dir, "train_stats.json")
        self.train_dataset.save_normalization_stats(self.stats_path)

    def set_input(self, sample):
        input, target,_,_,seg_mask,_  = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.seg_mask = seg_mask.to(self.device)
        #self.image_index = image_index
        
    def train(self):
        """
        Train stage.
        """
        best_model_full_path = None
        best_metric = 10000
        num_epoch_no_improvement = 0
        if self.part_of_pipeline:
            self.pipeline_monitor.set_metric_attributes(train_metric=self.config.loss + " loss",
                                                        valid_metric='mse loss')

        results = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.start_epoch, self.config.epochs):
            self.recorder.logger.info('Epoch: %d/%d lr %e', epoch, self.config.epochs,
                                      self.optimizer.param_groups[0]['lr'])
            self.model.train()
            self.network.train()

            tloss_r = AverageMeter()
            train_bar = tqdm(self.train_dataloader)
            for itr, sample in tqdm(enumerate(train_bar)):
                self.set_input(sample)
                self.optimize_parameters()
                tloss_r.update(self.loss.item(), self.input.size(0))
                train_bar.set_postfix(loss=tloss_r.avg)
                if (epoch == 1) and (itr == 1) and self.config.save_tensorboard_graph:
                    self.recorder.writer.add_graph(self.model, self.input)

            self.recorder.logger.info("Epoch {} , Train-loss:{:.3f}".format(epoch, tloss_r.avg))
            self.recorder.writer.add_scalar('Train/total_loss', tloss_r.avg, epoch)

            sys.stdout.flush()

            if epoch % self.config.val_freq == 0:
                self.recorder.logger.info("***************Validation at epoch {}****************".format(epoch))
                self.eval_dataset.load_normalization_stats(self.stats_path)
                vloss_r = AverageMeter()
                valid_bar = tqdm(self.eval_dataloader)
                self.model.eval()
                self.network.eval()
                with torch.no_grad():
                    for itr, sample in tqdm(enumerate(valid_bar)):
                        image, target, _,_ = sample[0], sample[1], sample[2], sample[3]
                        image = image.to(self.device)
                        target = target.to(self.device)
                        seg_mask = sample[4].to(self.device)
                        pred = self.model(image)
                        if self.config.loss == 'target_mask_weighted_mse':
                            val_loss = self.criterion(pred, target, seg_mask)
                        else:
                            val_loss = self.criterion(pred, target)
                        
                        vloss_r.update(val_loss.item(), image.size(0))
                        valid_bar.set_postfix(loss=vloss_r.avg)
                    self.recorder.logger.info("Epoch {} , Validation-loss:{:.3f}".format(epoch, vloss_r.avg))
                    self.recorder.writer.add_scalar('Validation/total_loss', vloss_r.avg, epoch)

                    if self.part_of_pipeline and self.pipeline_pretext_name is not None:
                        self.pipeline_monitor.pipeline_writer.add_scalar(
                            self.pipeline_pretext_name + "/Downstream/Mapping2D/" + self.config.network + "/Loss/train",
                            tloss_r.avg, epoch)
                        self.pipeline_monitor.pipeline_writer.add_scalar(
                            self.pipeline_pretext_name + "/Downstream/Mapping2D/" + self.config.network + "/Loss/valid",
                            vloss_r.avg, epoch)

                    results['train_loss'].append(tloss_r.avg)
                    results['val_loss'].append(vloss_r.avg)

                    data_frame = pd.DataFrame(data={'Train_Loss': results['train_loss'],
                                                    'val_loss': results['val_loss']},
                                            index=range(self.start_epoch, epoch + 1, self.config.val_freq))

                    data_frame.to_csv(os.path.join(self.recorder.save_dir, "results.csv"), index_label='epoch')

                    self.recorder.plot_loss(self.start_epoch, epoch + 1, self.config.val_freq, results['train_loss'])
                    self.recorder.plot_val_metrics(self.start_epoch, epoch + 1, self.config.val_freq, results['val_loss'])

                    # early stopping
                    if vloss_r.avg < best_metric:
                        self.recorder.logger.info(
                            "Validation metric decreases from {:.4f} to {:.4f}".format(best_metric, vloss_r.avg))
                        best_metric = vloss_r.avg
                        num_epoch_no_improvement = 0
                        best_model_full_path = os.path.join(self.recorder.save_dir, "best_model.pth")
                        self.save_state_dict(epoch, best_model_full_path)
                        self.recorder.logger.info(
                            "Saving the best model at epoch {} to '{}'".format(epoch, best_model_full_path))
                        if self.part_of_pipeline:
                            self.pipeline_monitor.set_best_model_trackers(train_loss=tloss_r.avg, valid_perf=vloss_r.avg,
                                                                        stop_epoch=epoch + 1,
                                                                        best_model_path=best_model_full_path)
                    else:
                        self.recorder.logger.info(
                            "Validation metric does not decrease from {:.4f}, num_epoch_no_improvement {}".format(
                                best_metric,
                                num_epoch_no_improvement))
                        num_epoch_no_improvement += 1

                        if (self.fine_tuning_scheme == 'warmup_by_layer_by_patience') \
                                and (len(self.encoder_layers_reversed)>0) \
                                and (num_epoch_no_improvement == self.config.warmup_patience):
                            self.recorder.logger.info("Unfreeze Layer")
                            self.check_freezing_patience_gradual_step()
                            print("reinitalizing num_epoch_no_improvement to 0 because of unfreezing an encoder block")
                            num_epoch_no_improvement = 0

                        if (num_epoch_no_improvement == self.config.patience):
                            self.recorder.logger.info("Early Stopping")
                            break
            if self.scheduler is not None:
                if self.config.scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(vloss_r.avg)
                else:
                    self.scheduler.step()

        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return

    def post_processing(self, pred_seg):
        pass
