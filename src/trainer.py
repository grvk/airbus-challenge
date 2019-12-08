from os import path, makedirs
from datetime import datetime
from dill.source import getsource
from timeit import default_timer as timer
import torch

class Trainer(object):
    """
    Class simlifying training process of neural nets

    Trainer helps to manage periodic comprehensive backups of the
    model, trainer, loss functions, optimizer etc. Network training
    could be continued from the point of any backup without losing
    any state data. Additionally, the class provides such information
    as: average epoch runtime, average batch runtime, predictions for
    the training process time, live loss values (on train and
    vavalidation sets), loss values' trends (increasing/decreasing).
    Finally, it stores loss values on the train and validation datasets
    for each epoch for later analysis, outputs final accuracy, and stores
    a final backup.

    Args:
        model(:obj:`torch.nn.Module`): model to be trained
        optimizer(:obj:`torch.optim.Optimizer`): optimizer function used to
            update weights (i.e. SGD or Adam)
        loss_fn(:obj:`function`, :obj:`torch.nn._Loss`): a loss function to
            be applied to the output of the model during training
        final_eval_fn(function, optional): function that will run once training
            is done to evaluate performance on train and validation sets.
            Accepts expected output from dataloader and output from the model.
            Example: classification accuracy or IoU for segmentation.
            Nothing is run by default.
        train_dataloader_creator(:obj:`function`): function returning a
            new instance of a dataloader on a train dataset
        val_dataloader_creator(:obj:`function`): function returning a
            new instance of a dataloader on a validation dataset
        backup_interval(int, optional): epoch interval after which neural net
            is backed up. Defaults to 10. If set, this argument has priority
            over backup_interval provided as part of trainer_state
        device(:obj:`torch.device`, optional): device, on which neural net will
            be trained. Should be the same device, on which model runs.
            NOTE: send the model to this device before initializing an
            optimizer. Defaults to torch.device("cpu").
        custom_back_up_path(string, optional): path to the directory,
            where backups will be stored. Defaults to airbus-challenge/backups
        is_debug_mode(bool): if in debug mode, autograd.detect_anomaly()
            hook will be used to make sure no tensers become nan's
        additional_backup_info(dict): any extra info that needs to be backed up
        trainer_state(:obj:`dict`, optional): trainer state to initialize with.
            Usually, comes as part of the backup. See _back_up() for more
            details. By default, provides clean state.

    Attributes:
        device(:obj:`torch.device`): device where neural net is being trained.
        model(:obj:`torch.nn.Module`): model that is being trained
        train_loss_history(:obj:`list[float]`): loss values on train set for
            each epoch. Calculated as a runtime average
        validation_loss_history(:obj:`list[float]`): loss values on validation
            set. Calculated at the end of each epoch.
        final_train_eval(float): reported evaluation on train set at the end of the
            training (if requested)
        final_val_eval(float): reported evaluation on validation set at the end of
            the training (if requested)
    """

    DEFAULT_BACK_UP_PATH = path.join(path.dirname(__file__), '../backups/')
    DEFAULT_BACK_UP_INTERVAL = 10
    DEFAULT_DEVICE = torch.device('cpu')

    def __init__(self, model, optimizer, loss_fn,
            train_dataloader_creator, val_dataloader_creator,
            backup_interval=None, device=DEFAULT_DEVICE, final_eval_fn=None,
            custom_back_up_path = None, is_debug_mode=False,
            additional_backup_info = None, trainer_state={}):

        if is_debug_mode:
            torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.final_eval_fn = final_eval_fn

        self.train_dataloader_creator = train_dataloader_creator
        self.val_dataloader_creator = val_dataloader_creator

        # preference to explicit backup_interval setting
        # if not set, try to get one from the trainer_state
        # if not available, use default = 10
        self.backup_interval = backup_interval
        if self.backup_interval is None:
            self.backup_interval = trainer_state.get(
            'backup_interval', Trainer.DEFAULT_BACK_UP_INTERVAL)

        self.additional_backup_info = additional_backup_info

        self.cur_epoch_idx = trainer_state.get('cur_epoch_idx', 0)
        self.cur_train_loss = trainer_state.get("cur_train_loss", -1)
        self.cur_val_loss = trainer_state.get("cur_val_loss", -1)
        self.train_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.val_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.final_train_eval = trainer_state.get("final_train_eval", None)
        self.final_val_eval = trainer_state.get("final_val_eval", None)

        self.back_up_path = custom_back_up_path
        if self.back_up_path is None:
            self.back_up_path = trainer_state.get(
            "back_up_path", Trainer.DEFAULT_BACK_UP_PATH)

        self.number_of_batches_per_epoch = None

        self.train_loss_history = trainer_state.get('train_loss_history', [])
        self.validation_loss_history = \
            trainer_state.get('validation_loss_history', [])

        if not path.exists(self.back_up_path):
            makedirs(self.back_up_path)

    def _back_up(self):

        fnl_ev = self.final_eval_fn

        backup = {
            'states' : {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'fn_strings': {
                'loss_fn': getsource(self.loss_fn),
                'final_eval_fn': getsource(fnl_ev) if fnl_ev is not None else None,
                'train_dataloader_creator': \
                    getsource(self.train_dataloader_creator),
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
                'final_train_eval': self.final_train_eval,
                'final_val_eval': self.final_val_eval,
                'back_up_path': self.back_up_path
            },
            'additional_info': self.additional_backup_info
        }

        torch.save(backup, path.join(self.back_up_path,
            "{d}_epoch_{e}.pth".format(
            d=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
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
            # simple moving average over the last 5 values
            previous_sma = sum(h[-6:-1]) / 5
            new_sma = sum(h[-5:]) / 5
        else:
            # simple moving average over the len(losses) - 1 values
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
                loss = loss / len(input)
                val_loss_total += loss
                val_batches += 1

        self.model.train(init_model_state)
        return val_loss_total / val_batches

    def _train_epoch(self):
        assert self.model.training, "Model not in training mode"

        dtloadr = self.train_dataloader_creator()
        self.cur_train_loss = 0

        start_time = timer()
        end_time = None
        avg_time = 0
        avg_exec_time = 0

        for batch_idx, (input, expected_output) in enumerate(dtloadr):

            input = input.to(self.device)
            expected_output = expected_output.to(self.device)

            self.optimizer.zero_grad()

            actual_output = self.model(input)
            loss = self.loss_fn(actual_output, expected_output)
<<<<<<< HEAD
            loss = loss.div(len(input))
=======
>>>>>>> trainer implementation
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            # average
            self.cur_train_loss = self.cur_train_loss * \
                (batch_idx / (batch_idx + 1)) + loss_value / (batch_idx + 1)

            # benchmark and print every 50 batches
            if (batch_idx + 1) % 50 == 0:
                end_time = timer()
                exec_time_50_batches = end_time - start_time
                exec_time = exec_time_50_batches / 50.0

                avg_ratio = (batch_idx - 49) / (batch_idx + 1)
                avg_exec_time = avg_exec_time * avg_ratio + \
                    exec_time_50_batches / (batch_idx + 1)

                total_batches_str = self.number_of_batches_per_epoch or "?"

                n = self.number_of_batches_per_epoch
                if n is not None:
                    time_left = avg_exec_time * (n - batch_idx)
                    metrics = " Estimated time left={:.3}s.".format(time_left)
                else:
                    metrics = ""

                log_line = "   < {i}/{t} >   Cur avg train loss={l:.6f}. " \
                    "Iteration exec time={at:.3}s.{m}".format(
                    i=batch_idx, t=total_batches_str, at=exec_time,
                    m=metrics, l=self.cur_train_loss
                )
                print(log_line)
                start_time = end_time

        self.cur_val_loss = self._get_validation_set_loss()

        self.train_loss_history.append(self.cur_train_loss)
        self.validation_loss_history.append(self.cur_val_loss)

        self._update_train_loss_diff()
        self._update_val_loss_diff()

    def evaluate_performance(self, dataloader_fn):
        """Run model on the whole dataset and return evaluated performance"""
        init_model_state = self.model.training
        self.model.eval()

        dtloadr = dataloader_fn()

        performance = None
        with torch.no_grad():

            total_metric = 0
            count = 0
            for (input, expected_output) in dtloadr:
                input = input.to(self.device)
                expected_output = expected_output.to(self.device)
                actual_output = self.model(input)

                metric = self.final_eval_fn(actual_output, expected_output)
                total_metric += metric
                count += 1

            performance = total_metric / count

        self.model.train(init_model_state)
        return performance

    def train(self, epochs_num):
        """
        Train model for the given number of epochs.

        The method shows a progress bar, which reports loss on the train
        dataset, validatiton dataset, and trends of these losses. The values
        are reported for the last epoch. Trends are represented with simple
        moving averages: difference in average loss values over the last 5
        epochs and over the 5 epochs preceding the last epoch. Negative values
        indicate decreasing trend (loss values decrease). To calculate the
        trends, the neural net needs to be trained at least for 2 epochs.
        By default the value is -1. At the end of the training a backup is
        created and accuracies on the train and validation sets are returned.
        Additionally, model is backed up every self.backup_interval epochs

        Args:
            epochs_num(int): number of epochs to train for

        Returns:
            tuple(float, float): model accuracies on the train and validation
                set correspondingly
        """
        assert epochs_num > 0, "Cannot train for <= 0 epochs: {}".format(epochs_num)
        print("Training for {} epochs".format(epochs_num))
        print("Backing up results every {} epochs".format(self.backup_interval))

        end_idx = self.cur_epoch_idx + epochs_num
        bkp_interval = self.backup_interval

        self.model.train()

        start_time = timer()
        end_time = 0
        avg_exec_time = 0

        i = 0
        while i < epochs_num:
            self.number_of_batches_per_epoch = self._train_epoch()
            self.cur_epoch_idx += 1
            i += 1
            if bkp_interval and (self.cur_epoch_idx % bkp_interval == 0):
                self._back_up()

            end_time = timer()
            exec_time = end_time - start_time

            avg_exec_time = avg_exec_time * ((i - 1.0) / i) + exec_time / i
            time_left = avg_exec_time * (epochs_num - i)

            log_line = "[ Epoch {ce}/{te} ]   Train loss={tl:.6f}. " \
                "Diff={td:.6f}. Val loss={vl:.6f}. Diff={vd:.6f}. " \
                "Exec time={etm:.1f}s. Estimated time left={tml:.1f}s".format(
                tl=self.cur_train_loss, td=self.train_loss_diff,
                vl=self.cur_val_loss, vd=self.val_loss_diff,
                ce=(self.cur_epoch_idx - 1), te=(end_idx - 1),
                etm=exec_time, tml=time_left)


            print(log_line)
            start_time = end_time

        self.model.eval()

        if self.final_eval_fn is not None:
            self.final_train_eval = \
                self.evaluate_performance(self.train_dataloader_creator)
            self.final_val_eval = \
                self.evaluate_performance(self.val_dataloader_creator)
        self._back_up()

        print("Done. Train performance = {ta}. Validation performance = {va}".
            format(ta = self.final_train_eval, va = self.final_val_eval))

        return (self.final_train_eval, self.final_val_eval)
