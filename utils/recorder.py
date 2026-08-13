from tensorboardX import SummaryWriter
import os.path
import os
import time
import utils.tools as tools
import matplotlib.pyplot as plt
from pathlib import Path

class Recorder:
    def __init__(self, config):
        self.config = config
        # result_dir: checkpoints/phase/task/dataset_name/network_note/timestamp/
        if config.run_mode == 'individual_run_mode':
            self.save_dir = '../../checkpoints' \
                            + '/{}/'.format(config.run_mode) \
                            + '/{}/'.format(config.phase) \
                            + '/{}/'.format(config.task) \
                            + '/{}/'.format(config.dataset_name) \
                            + '/{}/'.format(config.network + '_' + config.note) \
                            + '/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        else:
                self.save_dir = 'checkpoints' \
                                + '/{}/'.format(config.run_mode) \
                                + '/{}/'.format(config.run_id) \
                                + '/{}/'.format(config.phase+'_'+config.dataset_name) \
                                + '/{}/'.format(config.task) \
                                + '/{}/'.format(config.network + '_' + config.note) \

        relative_path = Path(self.save_dir)
        absolute_path = relative_path.resolve()
        print(absolute_path)
        self.logger = tools.get_logger(self.save_dir)
        print('RUNDIR: {}'.format(self.save_dir))
        self.logger.info('{}-Train'.format(self.config.task))
        self.config.display(self.logger)

        # record config
        self.save_tbx_log = self.save_dir + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def logger_shutdown(self):
        import logging
        logging.shutdown()

    def plot_loss(self, start_epoch, epochs, val_freq, train_loss, flag='train'):
        # Draw the training loss.
        x1 = range(start_epoch, epochs, val_freq)
        y1 = train_loss
        plt.plot(x1, y1, '-')
        if flag == 'train':
            plt.title('Training loss vs.epochs')
            plt.xlabel('epoch')
            plt.ylabel('Training loss')
            # plt.show()
            plt.savefig(self.save_dir + '/training_loss.jpg')
        else:
            plt.title('Validation loss vs.epochs')
            plt.xlabel('epoch')
            plt.ylabel('Validation loss')
            # plt.show()
            plt.savefig(self.save_dir + '/validation_loss.jpg')

        # get current figure
        fig = plt.gcf()
        plt.close(fig)

        return

    def plot_val_metrics(self, start_epoch, epochs, val_freq, metrics):
        # Draw the validation metrics.
        x1 = range(start_epoch, epochs, val_freq)
        y1 = metrics
        plt.plot(x1, y1, '-')
        plt.title('Validation results vs.epochs')
        plt.xlabel('epoch')
        plt.ylabel('Validation metric')
        # plt.show()
        plt.savefig(self.save_dir + '/validation_results.jpg')

        # get current figure
        fig = plt.gcf()
        plt.close(fig)

        return

    def save_images(self, input, pred, label, image_index, save_path):
        # Save the images during training or testing stage.
        tools.save_np2nii(input, save_path, 'image' + image_index)
        tools.save_np2nii(pred, save_path, 'pre' + image_index)
        tools.save_np2nii(label, save_path, 'label' + image_index)





