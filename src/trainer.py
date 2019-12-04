from tqdm import tqdm, trange
from os import path, makedirs
from datetime import datetime
from dill.source import getsource
import torch

class Trainer(object):

    DEFAULT_BACK_UP_PATH = path.join(path.dirname(__file__), '../backups/')

    def __init__(self, model, optimizer, loss_fn, \
            train_dataloader_creator, val_dataloader_creator, \
            backup_interval=None, device=torch.device("cpu"), \
            custom_back_up_path = None, trainer_state={}):
        """

        """
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_dataloader_creator = train_dataloader_creator
        self.val_dataloader_creator = val_dataloader_creator

        # preference to excplicit backup_interval setting
        # if not set, try to get one from the trainer_state
        # if not available, use default = 10
        self.backup_interval = backup_interval
        if self.backup_interval is None:
            self.backup_interval = trainer_state.get('backup_interval', 10)

        self.cur_epoch_idx = trainer_state.get('cur_epoch_idx', -1)
        self.cur_train_loss = trainer_state.get("cur_train_loss", -1)
        self.cur_val_loss = trainer_state.get("cur_val_loss", -1)
        self.train_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.val_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.final_train_acc = trainer_state.get("final_train_acc", -1)
        self.final_val_acc = trainer_state.get("final_val_acc", -1)

        self.back_up_path = custom_back_up_path
        if self.back_up_path is None:
            self.back_up_path = trainer_state.get( \
                "back_up_path", Trainer.DEFAULT_BACK_UP_PATH)

        self.train_loss_history = trainer_state.get('train_loss_history', [])
        self.validation_loss_history = trainer_state.get('validation_loss_history', [])


        if not path.exists(self.back_up_path):
            makedirs(self.back_up_path)

    def _back_up(self):

        backup = {
            'states' : {
                'model': self.model.state_dict(),
                # type of optimizer ?
                # learning rate ? other options ?
                'optimizer': self.optimizer.state_dict()
            },
            'fn_strings': {
                'loss_fn': getsource(self.loss_fn),
                # dataset info
                'train_dataloader_creator': getsource(self.train_dataloader_creator),
                'val_dataloader_creator': getsource(self.val_dataloader_creator)
            },
            'trainer_state': {
                'train_loss_history': self.train_loss_history,
                'validation_loss_history': self.validation_loss_history,
                'backup_interval': self.backup_interval,
                'cur_epoch_idx': self.cur_epoch_idx,
                'cur_train_loss': self.cur_train_loss,
                'cur_val_loss': self.cur_val_loss,
                'train_loss_diff': self.train_loss_diff,
                'val_loss_diff': self.val_loss_diff,
                'final_train_acc': self.final_train_acc,
                'final_val_acc': self.final_val_acc,
                'back_up_path': self.back_up_path
            }
        }

        torch.save(backup, path.join(self.back_up_path, \
            "{d}_epoch_{e}.pth".format( \
            d=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), \
            e=self.cur_epoch_idx)))

    def _calculate_loss_diff(self, is_train):
        h = self.train_loss_history if is_train else self.validation_loss_history
        l = len(h)

        # simple moving averages
        previous_sma = None
        new_sma = None

        if l < 2:
            # to have difference = -1
            previous_sma = 1
            new_sma = 0
        elif l > 5:
            previous_sma = sum(h[-6:-1]) / 5
            new_sma = sum(h[-5:]) / 5
        else:
            previous_sma = sum(h[:-1]) / (l - 1)
            new_sma = sum(h[1:]) / (l - 1)

        return new_sma - previous_sma

    def _update_val_loss_diff(self):
        self.val_loss_diff = self._calculate_loss_diff(False)

    def _update_train_loss_diff(self):
        self.train_loss_diff = self._calculate_loss_diff(True)

    def _get_validation_set_loss(self):
        init_model_state = self.model.training
        self.model.eval()

        dtloadr = self.val_dataloader_creator()

        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for (input, expected_output) in dtloadr:
                input = input.to(self.device)
                expected_output = expected_output.to(self.device)

                actual_output = self.model(input)
                loss = self.loss_fn(actual_output, expected_output).item()
                val_loss_total += loss
                val_batches += 1

        self.model.train(init_model_state)
        return val_loss_total / val_batches

    def _get_validation_set_accuracy(self):
        init_model_state = self.model.training
        self.model.eval()

        dtloadr = self.val_dataloader_creator()
        val_acc = -1

        with torch.no_grad():
            correct = 0
            total = 0
            for (input, expected_output) in dtloadr:
                input = input.to(self.device)
                expected_output = expected_output.to(self.device)

                actual_output = self.model(input)
                predictions = torch.argmax(actual_output.data, 1)
                correct += (predictions == expected_output).sum().item()
                total += expected_output.size(0)

            val_acc = correct / total

        self.model.train(init_model_state)
        return val_acc

    def _get_train_set_accuracy(self):
        init_model_state = self.model.training
        self.model.eval()

        dtloadr = self.train_dataloader_creator()
        train_acc = -1

        with torch.no_grad():
            correct = 0
            total = 0

            for (input, expected_output) in dtloadr:
                input = input.to(self.device)
                expected_output = expected_output.to(self.device)

                actual_output = self.model(input)
                predictions = torch.argmax(actual_output.data, 1)
                correct += (predictions == expected_output).sum().item()
                total += expected_output.size(0)

            train_acc = correct / total

        self.model.train(init_model_state)
        return train_acc

    def _train_epoch(self):
        assert self.model.training, "Model not in training mode"

        dtloadr = self.train_dataloader_creator()


        for batch_idx, (input, expected_output) in tqdm(enumerate(dtloadr), leave=False):

            input = input.to(self.device)
            expected_output = expected_output.to(self.device)

            self.optimizer.zero_grad()

            actual_output = self.model(input)
            loss = self.loss_fn(actual_output, expected_output)
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            self.cur_train_loss = self.cur_train_loss * \
                (batch_idx / (batch_idx + 1)) + loss_value / (batch_idx + 1)

        self.cur_val_loss = self._get_validation_set_loss()

        self.train_loss_history.append(self.cur_train_loss)
        self.validation_loss_history.append(self.cur_val_loss)

        self._update_train_loss_diff()
        self._update_val_loss_diff()

    def train(self, epochs_num):
        assert epochs_num > 0, "Cannot train for <= 0 epochs: {}".format(epochs_num)
        print("Training for {} epochs".format(epochs_num))
        print("Backing up results every {} epochs".format(self.backup_interval))

        start_epoch_idx = self.cur_epoch_idx + 1
        bkp_interval = self.backup_interval

        self.model.train()

        # progress bar
        pbar = trange(start_epoch_idx, start_epoch_idx + epochs_num, ncols=80)

        for cur_epoch in pbar:
            self.cur_epoch_idx = cur_epoch
            pbar.set_description("Train loss={tl}, diff={td}. " \
                "Val loss={vl}, diff={vd}.".format( \
                tl=self.cur_train_loss, td=self.train_loss_diff, \
                vl=self.cur_val_loss, vd=self.val_loss_diff))
            self._train_epoch()

            if bkp_interval and ((cur_epoch + 1) % bkp_interval == 0):
                self._back_up()

        self.model.eval()

        self.final_train_acc = self._get_train_set_accuracy()
        self.final_val_acc = self._get_validation_set_accuracy()
        self._back_up()

        print("Done training. Train Accuracy = {ta}. Validation accuracy = {va}".
            format(ta = self.final_train_acc, va = self.final_val_acc))
