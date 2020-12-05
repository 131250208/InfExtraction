from InfExtraction.work_flows.utils import DefaultLogger
from InfExtraction.modules.preprocess import Preprocessor
import os
import torch
import wandb
from pprint import pprint
from tqdm import tqdm
from glob import glob
import time
import re


class Trainer:
    def __init__(self,
                 model,
                 device,
                 optimizer,
                 trainer_config,
                 logger):
        self.model = model
        self.tagger = model.tagger
        self.device = device
        self.optimizer = optimizer
        self.logger = logger

        self.run_name = trainer_config["run_name"]
        self.exp_name = trainer_config["exp_name"]
        self.logger_interval = trainer_config["log_interval"]

        scheduler_config = trainer_config["scheduler_config"]
        self.scheduler_name = scheduler_config["name"]
        self.scheduler = None

        if self.scheduler_name == "CAWR":
            T_mult = scheduler_config["T_mult"]
            rewarm_steps = scheduler_config["rewarm_steps"]
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, rewarm_steps, T_mult)

        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)

        elif self.scheduler_name == "Step":
            decay_rate = scheduler_config["decay_rate"]
            decay_steps = scheduler_config["decay_steps"]
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    # train step
    def train_step(self, batch_train_data):
        golden_tags = batch_train_data["golden_tags"]
        golden_tags = [tag.to(self.device) for tag in golden_tags]

        del batch_train_data["sample_list"]
        del batch_train_data["golden_tags"]
        for k, v in batch_train_data.items():
            batch_train_data[k] = v.to(self.device)

        pred_outputs = self.model(**batch_train_data)

        metrics_dict = self.model.get_metrics(pred_outputs, golden_tags)
        loss = metrics_dict["loss"]

        # bp
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics_dict

    # # valid step
    # def valid_step(self, batch_valid_data):
    #     sample_list = batch_valid_data["sample_list"]
    #     golden_tags = batch_valid_data["golden_tags"]
    #     golden_tags = [tag.to(self.device) for tag in golden_tags]
    #
    #     del batch_valid_data["sample_list"]
    #     del batch_valid_data["golden_tags"]
    #     for k, v in batch_valid_data.items():
    #         batch_valid_data[k] = v.to(self.device)
    #
    #     with torch.no_grad():
    #         pred_outputs = self.model(**batch_valid_data)
    #     pred_tag = (pred_outputs > 0.).long()
    #     metrics_dict = self.model.get_metrics(pred_outputs, golden_tags)
    #
    #     pred_sample_list = self.tagger.decode_batch(sample_list, pred_tag)
    #
    #     cpg_dict = self.model.metrics_cal.get_event_cpg_dict(pred_sample_list, sample_list)
    #     return seq_acc.item(), cpg_dict

    def train(self, dataloader, ep, num_epoch):
        # train
        self.model.train()
        t_ep = time.time()
        # avg_loss, total_loss, avg_seq_acc, total_seq_acc = 0., 0., 0., 0.
        fin_metrics_dict = {}
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

            batch_print_format = "\rexp: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, {}" + \
                                 "lr: {:.5}, batch_time: {:.5}, total_time: {:.5} -------------"

            print(batch_print_format.format(self.exp_name, self.run_name,
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

    # def valid(self, dataloader):
    #     # valid
    #     self.model.eval()
    #
    #     t_ep = time.time()
    #     total_sample_acc = 0.
    #     total_cpg_dict = {}
    #     for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="validating")):
    #         sample_acc, cpg_dict = self.valid_step(batch_valid_data)
    #         total_sample_acc += sample_acc
    #
    #         # init total_cpg_dict
    #         for k in cpg_dict.keys():
    #             if k not in total_cpg_dict:
    #                 total_cpg_dict[k] = [0, 0, 0]
    #
    #         for k, cpg in cpg_dict.items():
    #             for idx, n in enumerate(cpg):
    #                 total_cpg_dict[k][idx] += cpg[idx]
    #
    #     avg_sample_acc = total_sample_acc / len(dataloader)
    #
    #     log_dict, final_score = None, None
    #     if self.task_type == "re":
    #         rel_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["rel_cpg"])
    #         ent_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["ent_cpg"])
    #         final_score = rel_prf[2]
    #         log_dict = {
    #             "val_shaking_tag_acc": avg_sample_acc,
    #             "val_rel_prec": rel_prf[0],
    #             "val_rel_recall": rel_prf[1],
    #             "val_rel_f1": rel_prf[2],
    #             "val_ent_prec": ent_prf[0],
    #             "val_ent_recall": ent_prf[1],
    #             "val_ent_f1": ent_prf[2],
    #             "time": time.time() - t_ep,
    #         }
    #     elif self.task_type == "ee":
    #         trigger_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0],
    #                                                                  total_cpg_dict["trigger_iden_cpg"][1],
    #                                                                  total_cpg_dict["trigger_iden_cpg"][2])
    #         trigger_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0],
    #                                                                   total_cpg_dict["trigger_class_cpg"][1],
    #                                                                   total_cpg_dict["trigger_class_cpg"][2])
    #         arg_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0],
    #                                                              total_cpg_dict["arg_iden_cpg"][1],
    #                                                              total_cpg_dict["arg_iden_cpg"][2])
    #         arg_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_class_cpg"][0],
    #                                                               total_cpg_dict["arg_class_cpg"][1],
    #                                                               total_cpg_dict["arg_class_cpg"][2])
    #         final_score = trigger_class_prf[2]
    #         log_dict = {
    #             "val_shaking_tag_acc": avg_sample_acc,
    #             "val_trigger_iden_prec": trigger_iden_prf[0],
    #             "val_trigger_iden_recall": trigger_iden_prf[1],
    #             "val_trigger_iden_f1": trigger_iden_prf[2],
    #             "val_trigger_class_prec": trigger_class_prf[0],
    #             "val_trigger_class_recall": trigger_class_prf[1],
    #             "val_trigger_class_f1": trigger_class_prf[2],
    #             "val_arg_iden_prec": arg_iden_prf[0],
    #             "val_arg_iden_recall": arg_iden_prf[1],
    #             "val_arg_iden_f1": arg_iden_prf[2],
    #             "val_arg_class_prec": arg_class_prf[0],
    #             "val_arg_class_recall": arg_class_prf[1],
    #             "val_arg_class_f1": arg_class_prf[2],
    #             "time": time.time() - t_ep,
    #         }
    #
    #     self.logger.log(log_dict)
    #     pprint(log_dict)
    #
    #     return final_score


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
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="predicting")):
            pred_sample_list = self._predict_step_debug(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
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
            batch_predict_data[k] = v.to(self.device)

        with torch.no_grad():
            pred_outputs = self.model(**batch_predict_data)

        if type(pred_outputs) == tuple:
            pred_tags = [self.model.pred_output2pred_tag(pred_out) for pred_out in pred_outputs]
        else:
            pred_tags = [self.model.pred_output2pred_tag(pred_outputs), ]

        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tags)
        return pred_sample_list

    def _alignment(self, pred_sample_list, golden_data):
        # decompose splits
        pred_sample_list = Preprocessor.decompose2splits(pred_sample_list)  # debug

        # merge and alignment
        id2text = {sample["id"]: sample["text"] for sample in golden_data}
        merged_pred_samples = {}
        for sample in pred_sample_list:
            id_ = sample["id"]
            # recover spans by offsets
            sample = Preprocessor.span_offset(sample, sample["tok_level_offset"], sample["char_level_offset"])
            # merge
            if id_ not in merged_pred_samples:
                merged_pred_samples[id_] = {
                    "id": id_,
                    "text": id2text[id_],
                    "entity_list": [],
                    "relation_list": [],
                    "event_list": [],
                }
            if "entity_list" in sample:
                merged_pred_samples[id_]["entity_list"].extend(sample["entity_list"])
            if "relation_list" in sample:
                merged_pred_samples[id_]["relation_list"].extend(sample["relation_list"])
            if "event_list" in sample:
                merged_pred_samples[id_]["event_list"].extend(sample["event_list"])

        # alignment (order)
        pred_data = []
        for sample in golden_data:
            id_ = sample["id"]
            pred_data.append(merged_pred_samples[id_])

        for sample in pred_data:
            if "entity_list" in sample:
                sample["entity_list"] = Preprocessor.unique_list(sample["entity_list"])
            if "relation_list" in sample:
                sample["relation_list"] = Preprocessor.unique_list(sample["relation_list"])
            if "event_list" in sample:
                sample["event_list"] = Preprocessor.unique_list(sample["event_list"])

        return pred_data

    def predict(self, dataloader, golden_data):
        # predict
        self.model.eval()
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="predicting")):
            pred_sample_list = self._predict_step(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        pred_data = self._alignment(total_pred_sample_list, golden_data)
        return pred_data
        
    def score(self, pred_data, golden_data, data_type):
        '''
        :param pred_data:
        :param golden_data:
        :param data_type: test or valid, for logging
        :param final_score_key: which score is the final score: trigger_class_f1, rel_f1
        :return:
        '''
        # clean extra info added by preprocessing
        def clean_data(data):
            for sample in data:
                sample["entity_list"] = [ent for ent in sample["entity_list"] if ent["type"].split(":")[0] != "EXT"]
        clean_data(pred_data)
        clean_data(golden_data)

        return self.model.metrics_cal.score(pred_data, golden_data, data_type)
