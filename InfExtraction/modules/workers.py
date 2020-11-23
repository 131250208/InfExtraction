from InfExtraction.work_flows.utils import DefaultLogger
import os
import torch
import wandb
from pprint import pprint
from tqdm import tqdm
import glob
import time


class Trainer:
    def __init__(self,
                 model,
                 device,
                 optimizer,
                 trainer_config,
                 logger,
                 model_state_dict_save_dir):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.max_score = 0.
        self.model_state_dict_save_dir = model_state_dict_save_dir

        self.run_name = trainer_config["run_name"]
        self.exp_name = trainer_config["exp_name"]
        self.score_threshold = trainer_config["score_threshold"]
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
    def train_step(self, batch_train_data):
        batch_shaking_tag = batch_train_data["shaking_tag"]
        batch_shaking_tag = batch_shaking_tag.to(self.device)

        del batch_train_data["sample_list"]
        del batch_train_data["shaking_tag"]
        for k, v in batch_train_data.items():
            batch_train_data[k] = v.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        pred_outputs = self.model(**batch_train_data)
        loss = self.model.get_loss(pred_outputs, batch_shaking_tag)

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

        pred_outputs = self.model(**batch_valid_data)
        pred_tag = (pred_outputs > 0.).long()
        seq_acc = self.model.metrics_cal.get_tag_seq_accuracy(pred_tag, batch_shaking_tag)

        pred_sample_list = self.model.tagger.decode_batch(sample_list, pred_tag)
        cpg_dict = self.model.metrics_cal.get_event_cpg_dict(pred_sample_list, sample_list)
        return seq_acc.item(), cpg_dict

    def train_n_valid(self, train_dataloader, valid_dataloader, num_epoch):
        def train(dataloader, ep):
            # train
            self.model.train()
            t_ep = time.time()
            avg_loss, total_loss, avg_seq_acc, total_seq_acc = 0., 0., 0., 0.
            for batch_ind, batch_train_data in enumerate(dataloader):
                t_batch = time.time()
                loss, seq_acc = self.train_step(batch_train_data)

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
                if type(self.logger) is wandb and batch_ind % self.logger_interval == 0:
                    self.logger.log(log_dict)
                elif type(self.logger) is DefaultLogger and (batch_ind + 1) == len(dataloader):
                    # if logger is not wandb, only log once at the end
                    self.logger.log(log_dict)

        def valid(dataloader):
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
            if "rel_cpg" in total_cpg_dict:
                rel_prf = self.model.metrics.get_prf_scores(*total_cpg_dict["rel_cpg"])
                ent_prf = self.model.metrics.get_prf_scores(*total_cpg_dict["ent_cpg"])
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
            elif "trigger_iden_cpg" in total_cpg_dict:
                trigger_iden_prf = self.model.metrics.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0],
                                                                     total_cpg_dict["trigger_iden_cpg"][1],
                                                                     total_cpg_dict["trigger_iden_cpg"][2])
                trigger_class_prf = self.model.metrics.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0],
                                                                      total_cpg_dict["trigger_class_cpg"][1],
                                                                      total_cpg_dict["trigger_class_cpg"][2])
                arg_iden_prf = self.model.metrics.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0],
                                                                 total_cpg_dict["arg_iden_cpg"][1],
                                                                 total_cpg_dict["arg_iden_cpg"][2])
                arg_class_prf = self.model.metrics.get_prf_scores(total_cpg_dict["arg_class_cpg"][0],
                                                                  total_cpg_dict["arg_class_cpg"][1],
                                                                  total_cpg_dict["arg_class_cpg"][2])
                final_score = arg_class_prf[2]
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

        for ep in range(num_epoch):
            train(train_dataloader, ep)
            fin_score = valid(valid_dataloader)

            if fin_score >= self.max_score:
                self.max_score = fin_score
                if fin_score > self.score_threshold:  # save the best model
                    modle_state_num = len(glob.glob(self.model_state_dict_save_dir + "/model_state_dict_*.pt"))
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_state_dict_save_dir, "model_state_dict_{}.pt".format(modle_state_num)))
                    print("Current score: {}, Best score: {}".format(fin_score, self.max_score))