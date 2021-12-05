from InfExtraction.modules.utils import DefaultLogger
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules.metrics import MetricsCalculator
import os
import torch
import wandb
from pprint import pprint
from tqdm import tqdm
from glob import glob
import time
import re
import copy

class Trainer:
    def __init__(self,
                 run_id,
                 model,
                 dataloader,
                 device,
                 optimizer,
                 trainer_config,
                 logger):
        self.run_id = run_id
        self.model = model
        self.tagger = model.tagger
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.dataloader = dataloader

        self.run_name = trainer_config["run_name"]
        self.exp_name = trainer_config["exp_name"]
        self.logger_interval = trainer_config["log_interval"]

        scheduler_config = trainer_config["scheduler_config"]
        self.scheduler_name = scheduler_config["name"]
        self.scheduler = None

        if self.scheduler_name == "CAWR":
            T_mult = scheduler_config["T_mult"]
            rewarm_steps = scheduler_config["rewarm_epochs"] * len(dataloader)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, rewarm_steps, T_mult)

        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)

        elif self.scheduler_name == "Step":
            decay_rate = scheduler_config["decay_rate"]
            decay_steps = scheduler_config["decay_steps"]
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    # train step
    def train_step(self, batch_train_data):
        del batch_train_data["sample_list"]

        def to_device_iter(inp_item):
            if type(inp_item) is dict:
                for k, v in inp_item.items():
                    if type(v) is torch.Tensor:
                        inp_item[k] = v.to(self.device)
                    else:
                        to_device_iter(v)
            elif type(inp_item) is list:
                for idx, v in enumerate(inp_item):
                    if type(v) is torch.Tensor:
                        inp_item[idx] = v.to(self.device)
                    else:
                        to_device_iter(v)
            else:
                pass

        to_device_iter(batch_train_data)
        golden_tags = batch_train_data["golden_tags"]
        del batch_train_data["golden_tags"]
        pred_outputs = self.model(**batch_train_data)

        metrics_dict = self.model.get_metrics(pred_outputs, golden_tags)
        loss = metrics_dict["loss"]

        # bp
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics_dict

    def train(self, ep, num_epoch):
        # train
        self.model.train()
        t_ep = time.time()

        fin_metrics_dict = {}
        dataloader = self.dataloader
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            metrics_dict = self.train_step(batch_train_data)

            # log
            log_dict = {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            }
            metrics_log = ""
            for key, met in metrics_dict.items():
                fin_metrics_dict[key] = fin_metrics_dict.get(key, 0) + met.item()
                avg_met = fin_metrics_dict[key] / (batch_ind + 1)
                metrics_log += "train_{}: {:.5}, ".format(key, avg_met)
                log_dict["train_{}".format(key)] = avg_met

            # scheduler
            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(fin_metrics_dict["loss"] / (batch_ind + 1))
            else:
                self.scheduler.step()

            batch_print_format = "\rrun_id: {}, exp: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, {}" + \
                                 "lr: {:.5}, batch_time: {:.5}, total_time: {:.5} -------------"

            print(batch_print_format.format(self.run_id, self.exp_name, self.run_name,
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            metrics_log,
                                            self.optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            if type(self.logger) is type(wandb) and batch_ind % self.logger_interval == 0:
                self.logger.log(log_dict)
            elif type(self.logger) is DefaultLogger and (batch_ind + 1) == len(dataloader):
                # if logger is not wandb, only log once at the end
                self.logger.log(log_dict)


class Evaluator:
    def __init__(self,
                 model,
                 device):
        self.model = model
        self.decoder = model.tagger
        self.device = device

    def _predict_step_debug(self, batch_predict_data):
        sample_list = batch_predict_data["sample_list"]
        pred_tags = batch_predict_data["golden_tags"]
        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tags)
        return pred_sample_list

    def _predict_debug(self, dataloader, golden_data):
        # predict
        self.model.eval()
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="debug: predicting")):
            pred_sample_list = self._predict_step_debug(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
        if hasattr(self.decoder, "trans"):
            pred_data = [self.decoder.trans(sample) for sample in pred_data]
        return pred_data

    def check_tagging_n_decoding(self, dataloader, golden_data):
        pred_data = self._predict_debug(dataloader, golden_data)
        return self.score(pred_data, golden_data, "debug")

    # predict step
    def _predict_step(self, batch_predict_data):
        sample_list = batch_predict_data["sample_list"]
        del batch_predict_data["sample_list"]
        if "golden_tags" in batch_predict_data:
            del batch_predict_data["golden_tags"]

        for k, v in batch_predict_data.items():
            if k == "padded_text_list":
                for sent in v:
                    sent.to(self.device)
            else:
                batch_predict_data[k] = v.to(self.device)

        with torch.no_grad():
            pred_outputs = self.model(**batch_predict_data)

        if type(pred_outputs) == tuple:
            pred_tags = [self.model.pred_output2pred_tag(pred_out) for pred_out in pred_outputs]
            pred_outputs = pred_outputs
        else:
            pred_tags = [self.model.pred_output2pred_tag(pred_outputs), ]
            pred_outputs = [pred_outputs, ]

        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tags, pred_outputs)
        return pred_sample_list

    def _alignment(self, pred_sample_list, golden_data):
        # decompose to splits
        pred_sample_list = Preprocessor.decompose2splits(pred_sample_list)

        # merge and alignment
        merged_pred_res = {}
        res_keys = {"entity_list", "relation_list", "event_list", "open_spo_list"}

        # merge
        for sample in pred_sample_list:
            id_ = sample["id"]
            # recover spans by offsets
            sample = Preprocessor.span_offset(sample, sample["tok_level_offset"], sample["char_level_offset"])

            # merge
            merged_pred_res.setdefault(id_, {})
            for key in res_keys:
                if key in sample:
                    merged_pred_res[id_].setdefault(key, []).extend(sample[key])

        # align by ids (in order)
        pred_data = []
        # pseudo_res = {key: [] for key in res_keys}
        for sample in golden_data:
            id_ = sample["id"]
            pred_res = merged_pred_res[id_]
            pred_sample = {
                "id": sample["id"],
                "text": sample["text"],
                "features": sample["features"],
                **pred_res
            }
            if id_ not in merged_pred_res:
                pred_sample["not_in_merged_res"] = True
            pred_data.append(pred_sample)

        for sample in pred_data:
            for key in res_keys:
                if key in sample:
                    sample[key] = Preprocessor.unique_list(sample[key])
        return pred_data

    def predict(self, dataloader, golden_data):
        # predict
        self.model.eval()
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="predicting")):
            pred_sample_list = self._predict_step(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
        if hasattr(self.decoder, "trans"):
            pred_data = [self.decoder.trans(sample) for sample in pred_data]
        return pred_data
        
    def score(self, pred_data, golden_data, data_filename=""):
        return MetricsCalculator.score(pred_data, golden_data, data_filename)
