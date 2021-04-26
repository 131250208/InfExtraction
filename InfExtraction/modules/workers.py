from InfExtraction.modules.utils import DefaultLogger
from InfExtraction.modules.preprocess import Preprocessor
import torch
import wandb
from tqdm import tqdm
import time


class Trainer:
    def __init__(self,
                 model,
                 dataloader,
                 device,
                 optimizer,
                 trainer_config,
                 logger):
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
        golden_tags = batch_train_data["golden_tags"]
        golden_tags = [tag.to(self.device) for tag in golden_tags]

        del batch_train_data["sample_list"]
        del batch_train_data["golden_tags"]
        for k, v in batch_train_data.items():
            if k == "padded_text_list":
                for sent in v:
                    sent.to(self.device)
            else:
                batch_train_data[k] = v.to(self.device)

        pred_outputs = self.model(**batch_train_data)
        if type(pred_outputs) is not tuple:  # make it a tuple
            pred_outputs = (pred_outputs, )

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
        # avg_loss, total_loss, avg_seq_acc, total_seq_acc = 0., 0., 0., 0.
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
        for batch_ind, batch_predict_data in enumerate(tqdm(dataloader, desc="debug: predicting")):
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
        id2text = {sample["id"]: sample["text"] for sample in golden_data}
        merged_pred_samples = {}
        for sample in pred_sample_list:
            id_ = sample["id"]
            # recover spans by offsets
            anns = Preprocessor.span_offset(sample, sample["tok_level_offset"], sample["char_level_offset"])
            sample = {**sample, **anns}

            # merge
            if id_ not in merged_pred_samples:
                merged_pred_samples[id_] = {
                    "id": id_,
                    "text": id2text[id_],
                }
            if "entity_list" in sample:
                if "entity_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["entity_list"] = []
                merged_pred_samples[id_]["entity_list"].extend(sample["entity_list"])
            if "relation_list" in sample:
                if "relation_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["relation_list"] = []
                merged_pred_samples[id_]["relation_list"].extend(sample["relation_list"])
            if "event_list" in sample:
                if "event_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["event_list"] = []
                merged_pred_samples[id_]["event_list"].extend(sample["event_list"])
            if "open_spo_list" in sample:
                if "open_spo_list" not in merged_pred_samples[id_]:
                    merged_pred_samples[id_]["open_spo_list"] = []
                # if id_ == 10467:
                #     print("de")
                merged_pred_samples[id_]["open_spo_list"].extend(sample["open_spo_list"])

        # alignment by id (in order)
        pred_data = []
        assert len(merged_pred_samples) == len(golden_data)
        for sample in golden_data:
            id_ = sample["id"]
            pred_data.append(merged_pred_samples[id_])

        def arg_to_str(arg):
            if len(arg["tok_span"]) > 0 and type(arg["tok_span"][0]) is list:
                arg_str = "-".join([arg["type"], arg["text"]])
            else:
                s = ",".join([str(idx) for idx in arg["tok_span"]]) if len(arg["tok_span"]) > 0 else arg["text"]
                arg_str = "-".join([arg["type"], s])
            return arg_str

        for sample in pred_data:
            if "entity_list" in sample:
                sample["entity_list"] = Preprocessor.unique_list(sample["entity_list"])
            if "relation_list" in sample:
                sample["relation_list"] = Preprocessor.unique_list(sample["relation_list"])
            if "event_list" in sample:
                sample["event_list"] = Preprocessor.unique_list(sample["event_list"])

                # filter redundant event
                new_event_list = []

                # if sample["id"] == 'ad6ce7c4f547c93efc184bd1c0b2a154':
                #     print("???!!!!")

                def to_e_set(event):
                    arg_list = event["argument_list"] + [{"type": "Trigger", "tok_span": event["trigger_tok_span"], "text": event["trigger"]}, ] \
                        if "trigger" in event else event["argument_list"]
                    event_set = {arg_to_str(arg) for arg in arg_list}
                    return event_set
                for e_i in sample["event_list"]:
                    event_set_i = to_e_set(e_i)
                    if any(event_set_i.issubset(to_e_set(e_j))
                           and len(event_set_i) < len(to_e_set(e_j))
                           for e_j in sample["event_list"]):  # if spo_i is a proper subset of another spo, skip
                        continue
                    new_event_list.append(e_i)
                sample["event_list"] = new_event_list

            if "open_spo_list" in sample:
                sample["open_spo_list"] = Preprocessor.unique_list(sample["open_spo_list"])

                # filter redundant spo
                new_spo_list = []
                for spo_i in sample["open_spo_list"]:
                    spo_set_i = {arg_to_str(arg) for arg in spo_i}
                    if any(spo_set_i.issubset({arg_to_str(arg) for arg in spo_j})
                           and len(spo_set_i) < len({arg_to_str(arg) for arg in spo_j})
                           for spo_j in sample["open_spo_list"]):  # if spo_i is a proper subset of another spo, skip
                        continue
                    new_spo_list.append(spo_i)
                sample["open_spo_list"] = new_spo_list

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
        
    def score(self, pred_data, golden_data, data_filename=""):
        '''
        :param pred_data:
        :param golden_data:
        :param data_filename: just for logging
        :param final_score_key: which score is the final score: trigger_class_f1, rel_f1
        :return:
        '''

        return self.model.metrics_cal.score(pred_data, golden_data, data_filename)
