import numpy as np
from numpy import inf
import torch
from torch import nn, optim
from utils.metrics import *
from utils.pytorchtools import *
from utils.util import *
from base.base_trainer import BaseTrainer
from logger.logger import *
from sklearn import metrics


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, model, criterion, optimizer, device, train_dataloader, valid_data):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.device = device

        self.train_dataloader = train_dataloader
        self.test_data = valid_data


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        all_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            batch = real_batch(batch)
            out = self.model(batch['item1'], batch['item2'], self.config['trainer']['task'])[0]
            loss = self.criterion(out, torch.FloatTensor(batch['label']).cuda())
            all_loss = all_loss + loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.save(self.model, './out/saved/models/KRED/checkpoint.pt')
        print("all loss: " + str(all_loss))


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        y_pred = []
        start_list = list(range(0, len(self.test_data['label']), int(self.config['data_loader']['batch_size'])))
        print("start_list len:", len(start_list))
        for start in start_list[:100]:
            print("start:", start)
            if start + int(self.config['data_loader']['batch_size']) <= len(self.test_data['label']):
                end = start + int(self.config['data_loader']['batch_size'])
            else:
                end = len(self.test_data['label'])
            out = self.model(self.test_data['item1'][start:end], self.test_data['item2'][start:end], self.config['trainer']['task'])[
                0].cpu().data.numpy()

            y_pred.extend(out)
        truth = self.test_data['label'][:len(y_pred)]
#        print("y_pred:", y_pred)
#        print("truth:", truth)
        nom=0
        denom=0
        for item_1, item_2 in zip(y_pred, truth):
            if item_1 <= 0.5 and item_2 == 0 or item_1 > 0.5 and item_2 == 1:
                nom += 1
            else:
                denom += 1
        acc_score = float(nom)/denom
        print("y_pred:", y_pred[:100])
        print("truth:", truth[:100])
                
        y_pred = np.array([item + 1 for item in y_pred])
        truth = np.array(truth)
        fpr, tpr, thresholds = metrics.roc_curve(truth, y_pred, pos_label=2)
        auc_score = metrics.auc(fpr, tpr)

#        auc_score = cal_auc(y_pred, truth)
        print("acc socre: " + str(acc_score))
        print("auc socre: " + str(auc_score))
        return auc_score

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state_model = self.model.state_dict()
        filename_model = str(self.checkpoint_dir / 'checkpoint-model-epoch{}.pth'.format(epoch))
        torch.save(state_model, filename_model)
        self.logger.info("Saving checkpoint: {} ...".format(filename_model))


    def train(self):
        """
            Full training logic
        """
        logger_train = get_logger("train")

        logger_train.info("model training")
        valid_scores = []
        early_stopping = EarlyStopping(patience=self.config['trainer']['early_stop'], verbose=True)
        for epoch in range(self.start_epoch, self.epochs+1):
            self._train_epoch(epoch)
            valid_socre = self._valid_epoch(epoch)
            valid_scores.append(valid_socre)
            early_stopping(valid_socre, self.model)
            if early_stopping.early_stop:
                logger_train.info("Early stopping")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

