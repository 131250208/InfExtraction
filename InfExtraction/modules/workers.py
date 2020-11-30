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
                 tagger,
                 device,
                 optimizer,
                 trainer_config,
                 logger):
        self.model = model
        self.tagger = tagger
        self.device = device
        self.optimizer = optimizer
        self.logger = logger

        self.task_type = trainer_config["task_type"]
        self.run_name = trainer_config["run_name"]
        self.exp_name = trainer_config["exp_name"]
        self.use_ghm = trainer_config["use_ghm"]
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
    def train_step(self, batch_train_data, ep):
        batch_shaking_tag = batch_train_data["shaking_tag"]
        batch_shaking_tag = batch_shaking_tag.to(self.device)

        del batch_train_data["sample_list"]
        del batch_train_data["shaking_tag"]
        for k, v in batch_train_data.items():
            batch_train_data[k] = v.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        pred_outputs = self.model(**batch_train_data)
        use_ghm = self.use_ghm if ep > 2 else False
        loss = self.model.get_loss(pred_outputs, batch_shaking_tag, use_ghm)

        loss.backward()
        self.optimizer.step()

        pred_tag = (pred_outputs > 0.).long()
        seq_acc = self.model.metrics_cal.get_tag_seq_accuracy(pred_tag, batch_shaking_tag)

        return loss.item(), seq_acc.item()

    # valid step
    def valid_step(self, batch_valid_data):
        sample_list = batch_valid_data["sample_list"]
        batch_shaking_tag = batch_valid_data["shaking_tag"]
        batch_shaking_tag = batch_shaking_tag.to(self.device)

        del batch_valid_data["sample_list"]
        del batch_valid_data["shaking_tag"]
        for k, v in batch_valid_data.items():
            batch_valid_data[k] = v.to(self.device)

        with torch.no_grad():
            pred_outputs = self.model(**batch_valid_data)
        pred_tag = (pred_outputs > 0.).long()
        seq_acc = self.model.metrics_cal.get_tag_seq_accuracy(pred_tag, batch_shaking_tag)

        pred_sample_list = self.tagger.decode_batch(sample_list, pred_tag)
        cpg_dict = self.model.metrics_cal.get_event_cpg_dict(pred_sample_list, sample_list)
        return seq_acc.item(), cpg_dict

    def train(self, dataloader, ep, num_epoch):
        # train
        self.model.train()
        t_ep = time.time()
        avg_loss, total_loss, avg_seq_acc, total_seq_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            loss, seq_acc = self.train_step(batch_train_data, ep)

            total_loss += loss
            total_seq_acc += seq_acc

            avg_loss = total_loss / (batch_ind + 1)

            # scheduler
            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            avg_seq_acc = total_seq_acc / (batch_ind + 1)

            batch_print_format = "\rexp: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + \
                                 "t_sample_acc: {}," + \
                                 "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(self.exp_name, self.run_name,
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            avg_loss,
                                            avg_seq_acc,
                                            self.optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            # log
            log_dict = {
                "train_loss": avg_loss,
                "train_small_shaking_seq_acc": avg_seq_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            }
            if type(self.logger) is type(wandb) and batch_ind % self.logger_interval == 0:
                self.logger.log(log_dict)
            elif type(self.logger) is DefaultLogger and (batch_ind + 1) == len(dataloader):
                # if logger is not wandb, only log once at the end
                self.logger.log(log_dict)

    def valid(self, dataloader):
        # valid
        self.model.eval()

        t_ep = time.time()
        total_sample_acc = 0.
        total_cpg_dict = {}
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="validating")):
            sample_acc, cpg_dict = self.valid_step(batch_valid_data)
            total_sample_acc += sample_acc

            # init total_cpg_dict
            for k in cpg_dict.keys():
                if k not in total_cpg_dict:
                    total_cpg_dict[k] = [0, 0, 0]

            for k, cpg in cpg_dict.items():
                for idx, n in enumerate(cpg):
                    total_cpg_dict[k][idx] += cpg[idx]

        avg_sample_acc = total_sample_acc / len(dataloader)

        log_dict, final_score = None, None
        if self.task_type == "re":
            rel_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["rel_cpg"])
            ent_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["ent_cpg"])
            final_score = rel_prf[2]
            log_dict = {
                "val_shaking_tag_acc": avg_sample_acc,
                "val_rel_prec": rel_prf[0],
                "val_rel_recall": rel_prf[1],
                "val_rel_f1": rel_prf[2],
                "val_ent_prec": ent_prf[0],
                "val_ent_recall": ent_prf[1],
                "val_ent_f1": ent_prf[2],
                "time": time.time() - t_ep,
            }
        elif self.task_type == "ee":
            trigger_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0],
                                                                     total_cpg_dict["trigger_iden_cpg"][1],
                                                                     total_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0],
                                                                      total_cpg_dict["trigger_class_cpg"][1],
                                                                      total_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0],
                                                                 total_cpg_dict["arg_iden_cpg"][1],
                                                                 total_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_class_cpg"][0],
                                                                  total_cpg_dict["arg_class_cpg"][1],
                                                                  total_cpg_dict["arg_class_cpg"][2])
            final_score = trigger_class_prf[2]
            log_dict = {
                "val_shaking_tag_acc": avg_sample_acc,
                "val_trigger_iden_prec": trigger_iden_prf[0],
                "val_trigger_iden_recall": trigger_iden_prf[1],
                "val_trigger_iden_f1": trigger_iden_prf[2],
                "val_trigger_class_prec": trigger_class_prf[0],
                "val_trigger_class_recall": trigger_class_prf[1],
                "val_trigger_class_f1": trigger_class_prf[2],
                "val_arg_iden_prec": arg_iden_prf[0],
                "val_arg_iden_recall": arg_iden_prf[1],
                "val_arg_iden_f1": arg_iden_prf[2],
                "val_arg_class_prec": arg_class_prf[0],
                "val_arg_class_recall": arg_class_prf[1],
                "val_arg_class_f1": arg_class_prf[2],
                "time": time.time() - t_ep,
            }

        self.logger.log(log_dict)
        pprint(log_dict)

        return final_score


class Evaluator:
    def __init__(self,
                 task_type,
                 model,
                 decoder,# tagger
                 device,
                 match_pattern=None):
        self.task_type = task_type
        self.model = model
        self.decoder = decoder
        self.device = device
        self.match_pattern = match_pattern # only for relation extraction

    # predict step
    def _predict_step(self, batch_predict_data):
        sample_list = batch_predict_data["sample_list"]
        del batch_predict_data["sample_list"]
        if "shaking_tag" in batch_predict_data:
            del batch_predict_data["shaking_tag"]
        for k, v in batch_predict_data.items():
            batch_predict_data[k] = v.to(self.device)

        with torch.no_grad():
            pred_outputs = self.model(**batch_predict_data)
        pred_tag = (pred_outputs > 0.).long()

        pred_sample_list = self.decoder.decode_batch(sample_list, pred_tag)
        return pred_sample_list
        
    def predict(self, dataloader, golden_data):
        # predict
        self.model.eval()
        total_pred_sample_list = []
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="predicting")):
            pred_sample_list = self._predict_step(batch_predict_data)
            total_pred_sample_list.extend(pred_sample_list)
        # decompose splits
        total_pred_sample_list = Preprocessor.decompose2splits(total_pred_sample_list) # debug

        # merge and alignment
        id2text = {sample["id"]:sample["text"] for sample in golden_data}
        merged_pred_samples = {}
        for sample in total_pred_sample_list:
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
        fin_pred_samples = []
        for sample in golden_data:
            id_ = sample["id"]
            fin_pred_samples.append(merged_pred_samples[id_])

        # # filter duplicated results
        # def unique(res_list):
        #     memory = set()
        #     new_res_list = []
        #     for res in res_list:
        #         if str(res) not in memory:
        #             new_res_list.append(res)
        #             memory.add(str(res))
        #     return new_res_list

        for sample in fin_pred_samples:
            if "entity_list" in sample:
                sample["entity_list"] = Preprocessor.unique_list(sample["entity_list"])
            if "relation_list" in sample:
                sample["relation_list"] = Preprocessor.unique_list(sample["relation_list"])
            if "event_list" in sample:
                sample["event_list"] = Preprocessor.unique_list(sample["event_list"])

        return fin_pred_samples

    def score(self, fin_pred_samples, golden_data, data_type, final_score_key):
        score_dict = None

        if self.task_type == "re":
            total_cpg_dict = self.model.metrics_cal.get_rel_cpg_dict(fin_pred_samples,
                                                                     golden_data,
                                                                     self.match_pattern)
            rel_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["rel_cpg"])
            ent_prf = self.model.metrics_cal.get_prf_scores(*total_cpg_dict["ent_cpg"])
            score_dict = {
                "{}_rel_prec".format(data_type): rel_prf[0],
                "{}_rel_recall".format(data_type): rel_prf[1],
                "{}_rel_f1".format(data_type): rel_prf[2],
                "{}_ent_prec".format(data_type): ent_prf[0],
                "{}_ent_recall".format(data_type): ent_prf[1],
                "{}_ent_f1".format(data_type): ent_prf[2]
            }
            score_dict["final_score"] = score_dict["{}_{}".format(data_type, final_score_key)]

        elif self.task_type == "ee":
            total_cpg_dict = self.model.metrics_cal.get_event_cpg_dict(fin_pred_samples, golden_data)
            trigger_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0],
                                                                     total_cpg_dict["trigger_iden_cpg"][1],
                                                                     total_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0],
                                                                      total_cpg_dict["trigger_class_cpg"][1],
                                                                      total_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0],
                                                                 total_cpg_dict["arg_iden_cpg"][1],
                                                                 total_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = self.model.metrics_cal.get_prf_scores(total_cpg_dict["arg_class_cpg"][0],
                                                                  total_cpg_dict["arg_class_cpg"][1],
                                                                  total_cpg_dict["arg_class_cpg"][2])

            score_dict = {
                "{}_trigger_iden_prec".format(data_type): trigger_iden_prf[0],
                "{}_trigger_iden_recall".format(data_type): trigger_iden_prf[1],
                "{}_trigger_iden_f1".format(data_type): trigger_iden_prf[2],
                "{}_trigger_class_prec".format(data_type): trigger_class_prf[0],
                "{}_trigger_class_recall".format(data_type): trigger_class_prf[1],
                "{}_trigger_class_f1".format(data_type): trigger_class_prf[2],
                "{}_arg_iden_prec".format(data_type): arg_iden_prf[0],
                "{}_arg_iden_recall".format(data_type): arg_iden_prf[1],
                "{}_arg_iden_f1".format(data_type): arg_iden_prf[2],
                "{}_arg_class_prec".format(data_type): arg_class_prf[0],
                "{}_arg_class_recall".format(data_type): arg_class_prf[1],
                "{}_arg_class_f1".format(data_type): arg_class_prf[2],
            }
            score_dict["final_score"] = score_dict["{}_{}".format(data_type, final_score_key)]

        return score_dict
