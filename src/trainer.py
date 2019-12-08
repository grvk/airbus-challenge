from tqdm import trange, tqdm
from os import path, makedirs
from datetime import datetime
from dill.source import getsource
import torch
from torch import autograd

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
    DEFAULT_DEVICE = torch.device("cpu")

    def __init__(self, model, optimizer, loss_fn, \
            train_dataloader_creator, val_dataloader_creator, \
            backup_interval=None, device=DEFAULT_DEVICE, final_eval_fn=None,\
            custom_back_up_path = None, extra_backup_info = None, trainer_state={}):
        
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
            self.backup_interval = trainer_state.get( \
            'backup_interval', Trainer.DEFAULT_BACK_UP_INTERVAL)
        
        self.extra_backup_info = extra_backup_info

        
        self.cur_epoch_idx = trainer_state.get('cur_epoch_idx', -1)
        self.cur_train_loss = trainer_state.get("cur_train_loss", -1)
        self.cur_val_loss = trainer_state.get("cur_val_loss", -1)
        self.train_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.val_loss_diff = trainer_state.get("train_loss_diff", -1)
        self.final_train_eval = trainer_state.get("final_train_eval", None)
        self.final_val_eval = trainer_state.get("final_val_eval", None)

        self.back_up_path = custom_back_up_path
        if self.back_up_path is None:
            self.back_up_path = trainer_state.get( \
            "back_up_path", Trainer.DEFAULT_BACK_UP_PATH)

        self.train_loss_history = trainer_state.get('train_loss_history', [])
        self.validation_loss_history = \
            trainer_state.get('validation_loss_history', [])

        if not path.exists(self.back_up_path):
            makedirs(self.back_up_path)

    def _back_up(self):
        backup = {
            'states' : {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
#             'fn_strings': {
#                 'loss_fn': getsource(self.loss_fn),
#                 'final_eval_fn': getsource(self.final_eval_fn),
#                 'train_dataloader_creator': \
#                     getsource(self.train_dataloader_creator),
#                 'val_dataloader_creator': getsource(self.val_dataloader_creator)
#             },
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
            'extra_info': self.extra_backup_info
        }

        torch.save(backup, path.join(self.back_up_path, \
            "{d}_epoch_{e}.pth".format( \
            d=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), \
            e=self.cur_epoch_idx)))
        
        del backup # ensuring cleanup

    def _calculate_loss_diff(self, is_train):
        h = self.train_loss_history if is_train else self.validation_loss_history
        l = len(h)
        
        if isinstance(h, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ h - tensor")
        if isinstance(l, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ l - tensor")

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
            if isinstance(previous_sma, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ previous_sma - tensor")
            if isinstance(new_sma, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ new_sma - tensor")
        else:
            # simple moving average over the len(losses) - 1 values
            previous_sma = sum(h[:-1]) / (l - 1)
            if isinstance(previous_sma, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ previous_sma - tensor")
            new_sma = sum(h[1:]) / (l - 1)
            if isinstance(new_sma, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ new_sma - tensor")

        return new_sma - previous_sma

    def _update_val_loss_diff(self):
        self.val_loss_diff = self._calculate_loss_diff(False)
        if isinstance(self.val_loss_diff, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ self.val_loss_diff - tensor")

    def _update_train_loss_diff(self):
        self.train_loss_diff = self._calculate_loss_diff(True)
        if isinstance(self.train_loss_diff, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ self.train_loss_diff - tensor")

    def _get_validation_set_loss(self):
        print("__get validation loss")
        init_model_state = self.model.training
        if isinstance(init_model_state, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ _get_validation_set_loss.init_model_state - tensor")
        self.model.eval()

        dtloadr = self.val_dataloader_creator()

        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for (input, expected_output) in dtloadr:
                device_input = input.to(self.device)
                device_expected_output = expected_output.to(self.device)
                
                actual_output = self.model(device_input)
                loss = self.loss_fn(actual_output, device_expected_output).item()
                print("LOSS={}".format(loss))
                
                if isinstance(loss, torch.Tensor):
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~ _get_validation_set_loss.loss - tensor")
                
                val_loss_total += loss
                val_batches += 1
                
                del input, device_input, expected_output, device_expected_output, actual_output

        self.model.train(init_model_state)
        if isinstance(val_loss_total, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ _get_validation_set_loss.val_loss_total - tensor")
            
        return val_loss_total / val_batches

    def _train_epoch(self):
        assert self.model.training, "Model not in training mode"

        dtloadr = self.train_dataloader_creator()
        self.cur_train_loss = 0

        with autograd.detect_anomaly():
            progress_bar = tqdm(enumerate(dtloadr), ncols=80, leave=False)
            #progress_bar = enumerate(dtloadr)
            for batch_idx, (input, expected_output) in progress_bar:

                device_input = input.to(self.device)
                device_expected_output = expected_output.to(self.device)

                self.optimizer.zero_grad()
                actual_output = self.model(device_input)
                loss = self.loss_fn(actual_output, device_expected_output) #??????

                if (batch_idx % 10) == 0:
                    self.train_loss_history.append(self.cur_train_loss)
                    self._update_train_loss_diff()
                    self._update_val_loss_diff()
                    progress_bar.set_description("Train loss={tl}, diff={td}. " \
                        "Val loss={vl}, diff={vd}.".format( \
                        tl=self.cur_train_loss, td=self.train_loss_diff, \
                        vl=self.cur_val_loss, vd=self.val_loss_diff))

                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                # average
                self.cur_train_loss = self.cur_train_loss * \
                    (batch_idx / (batch_idx + 1)) + loss_value / (batch_idx + 1)

                if isinstance(self.cur_train_loss, torch.Tensor):
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~ _train_epoch.self.cur_train_loss - tensor")

                del input, device_input, expected_output, device_expected_output, actual_output, loss

        

        self.cur_val_loss = self._get_validation_set_loss()

        self.train_loss_history.append(self.cur_train_loss)
        self.validation_loss_history.append(self.cur_val_loss)

        self._update_train_loss_diff()
        self._update_val_loss_diff()
        
        del dtloadr, progress_bar

    def evaluate_performance(self, dataloader_fn):
        """Run model on the whole dataset and return evaluated performance"""
        init_model_state = self.model.training
        
        if isinstance(init_model_state, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ evaluate_performance.init_model_state - tensor")
        
        self.model.eval()

        dtloadr = dataloader_fn()

        performance = None
        with torch.no_grad():

            total_metric = 0
            count = 0
            
            progress_bar = tqdm(dtloadr, leave=False, ncols=80)
            #progress_bar = dtloadr
            for (input, expected_output) in progress_bar:
                device_input = input.to(self.device)
                device_expected_output = expected_output.to(self.device)
                
                actual_output = self.model(device_input)

                metric = self.final_eval_fn(actual_output, device_expected_output)
                print("METRIC:  {}".format(metric))
                
                if isinstance(metric, torch.Tensor):
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~ evaluate_performance.metric - tensor")
                
                total_metric += metric
                count += 1
                
                del input, device_input, expected_output, device_expected_output, actual_output, metric

            performance = total_metric / count
            
            if isinstance(performance, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ evaluate_performance.performance - tensor") 

        self.model.train(init_model_state)
        
        del dtloadr, progress_bar
        
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

        #assert epochs_num > 0, "Cannot train for <= 0 epochs: {}".format(epochs_num)
        print("Training for {} epochs".format(epochs_num))
        print("Backing up results every {} epochs".format(self.backup_interval))

        start_epoch_idx = self.cur_epoch_idx + 1
        bkp_interval = self.backup_interval
        
        if isinstance(start_epoch_idx, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ train.start_epoch_idx - tensor") 
        
        if isinstance(bkp_interval, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ train.bkp_interval - tensor") 

        self.model.train()

        # progress bar
        #pbar = trange(start_epoch_idx, start_epoch_idx + epochs_num, ncols=80)
        pbar = range(start_epoch_idx, start_epoch_idx + epochs_num)

        for cur_epoch in pbar:
            self.cur_epoch_idx = cur_epoch
            if isinstance(self.cur_epoch_idx, torch.Tensor):
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~ train.self.cur_epoch_idx - tensor") 
            
#             pbar.set_description("Train loss={tl}, diff={td}. " \
#                 "Val loss={vl}, diff={vd}.".format( \
#                 tl=self.cur_train_loss, td=self.train_loss_diff, \
#                 vl=self.cur_val_loss, vd=self.val_loss_diff))

            self._train_epoch()
            self.cur_train_loss = 0
            self.cur_val_loss = 0

            #if bkp_interval and ((cur_epoch + 1) % bkp_interval == 0):
            #    self._back_up()

        self.model.eval()

        if self.final_eval_fn is not None:
            self.final_train_eval = \
                self.evaluate_performance(self.train_dataloader_creator)
            self.final_val_eval = \
                self.evaluate_performance(self.val_dataloader_creator)
        #self._back_up()

        if isinstance(self.final_val_eval, torch.Tensor):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~ train.self.final_val_eval - tensor") 
        
        print("Done. Train performance = {ta}. Validation performance = {va}".
            format(ta = self.final_train_eval, va = self.final_val_eval))

        del pbar
        
        return (self.final_train_eval, self.final_val_eval)
