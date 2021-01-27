import torch
import re
import copy
from IPython.core.debugger import set_trace
from InfExtraction.modules import utils
from torch import nn


class MetricsCalculator:
    def __init__(self,
                 use_ghm=False):

        # for multilabel_categorical_crossentropy
        self.use_ghm = use_ghm
        self.last_weights = None  # for exponential moving averaging

    def GHM(self, gradient, bins=10, beta=0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size_train, shaking_seq_len)
        '''
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid((gradient - avg) / std)  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(gradient_norm, 0, 0.9999999)  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar).float()) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        #         print()
        #         print("hits_vec: {}".format(hits_vec))
        #         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights
        #         print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(gradient_norm.size()[0], gradient_norm.size()[1], 1)
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    def multilabel_categorical_crossentropy(self, y_pred, y_true, bp_steps):
        """
        This function is a loss function for multi-label learning
        ref: https://kexue.fm/archives/7359

        y_pred: (batch_size_train, ... , type_size)
        y_true: (batch_size_train, ... , type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if self.use_ghm and bp_steps > 1000:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def bce_loss(self, y_pred, y_true):
        '''
        y_pred: (batch_size_train, ... , type_size)
        y_true: (batch_size_train, ... , type_size)
        :return: loss
        '''
        loss_func = nn.BCELoss()
        loss = loss_func(nn.Sigmoid()(y_pred), y_true.float())
        return loss

    def get_tag_seq_accuracy(self, pred, truth):
        '''
        the tag accuracy in a batch
        a predicted tag sequence (matrix) is correct if and only if the whole sequence is congruent to the golden sequence
        '''
        #         # (batch_size_train, ..., seq_len, tag_size) -> (batch_size_train, ..., seq_len)
        #         pred = torch.argmax(pred, dim = -1)
        # (batch_size_train, ..., seq_len) -> (batch_size_train, seq_len)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size_train, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc

    @staticmethod
    def get_mark_sets_ee(event_list):
        trigger_iden_set, trigger_class_set = set(), set()
        arg_hard_iden_set, arg_hard_class_set = set(), set()  # consider trigger offset
        arg_soft_iden_set, arg_soft_class_set = set(), set()  # do not consider trigger offset
        arg_link_iden_set, arg_link_class_set = set(), set()  # for trigger-free

        for event in event_list:
            # trigger-based metrics
            trigger_offset = None
            if "trigger" in event:
                event_type = event["trigger_type"]
                trigger_offset = event["trigger_tok_span"]
                trigger_iden_set.add(str(trigger_offset))
                trigger_class_set.add(str([event_type] + trigger_offset))

            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                event_type = arg["event_type"]

                arg_soft_iden_set.add(str([event_type] + argument_offset))
                arg_soft_class_set.add(str([event_type] + argument_offset + [argument_role]))
                if "trigger" in event:
                    assert trigger_offset is not None
                    arg_hard_iden_set.add(str([event_type] + argument_offset + trigger_offset))
                    arg_hard_class_set.add(str([event_type] + argument_offset + trigger_offset + [argument_role]))

            # trigger-free metrics
            arg_list = copy.deepcopy(event["argument_list"])
            if "trigger" in event and event["trigger"] != "":
                # take trigger as an ordinary argument
                arg_list.append({
                    "text": event["trigger"],
                    "tok_span": event["trigger_tok_span"],
                    "char_span": event["trigger_char_span"],
                    "type": "Trigger",  # argument role
                    "event_type": event["trigger_type"],
                })

            for arg_i in arg_list:
                for arg_j in arg_list:
                    arg_i_event_type = arg_i["event_type"]
                    arg_i_offset = arg_i["tok_span"]
                    arg_i_role = arg_i["type"]

                    arg_j_event_type = arg_j["event_type"]
                    arg_j_offset = arg_j["tok_span"]
                    arg_j_role = arg_j["type"]

                    link_iden_mark = str([arg_i_event_type] + [arg_j_event_type] + arg_i_offset + arg_j_offset)
                    link_class_mark = str([arg_i_event_type] + [arg_j_event_type] +
                                          arg_i_offset + arg_j_offset +
                                          [arg_i_role] + [arg_j_role])
                    arg_link_iden_set.add(link_iden_mark)
                    arg_link_class_set.add(link_class_mark)

        return {
            "trigger_iden": trigger_iden_set,
            "trigger_class": trigger_class_set,
            "arg_soft_iden": arg_soft_iden_set,
            "arg_soft_class": arg_soft_class_set,
            "arg_hard_iden": arg_hard_iden_set,
            "arg_hard_class": arg_hard_class_set,
            "arg_link_iden": arg_link_iden_set,
            "arg_link_class": arg_link_class_set,
        }

    @staticmethod
    def get_partial_ent(ent_text):
        ch_pattern = "[\u4e00-\u9fa5\s]+"
        try:
            part_ent = ent_text[0] if re.match(ch_pattern, ent_text[0]) is not None else ent_text.split()[0]
            return part_ent
        except Exception:
            set_trace()

    @staticmethod
    def get_mark_sets_ent(ent_list, sent_w_disc=False):
        ent_partial_text_set, \
        ent_partial_offset_set, \
        ent_exact_text_set, \
        ent_exact_offset_set, \
        disc_ent_exact_offset_set, \
        disc_ent_exact_text_set, \
        ent_exact_offset_on_sents_w_disc_set, \
        ent_exact_text_on_sents_w_disc_set = set(), set(), set(), set(), set(), set(), set(), set()

        for ent in ent_list:
            part_ent = MetricsCalculator.get_partial_ent(ent["text"])
            ent_partial_text_set.add(str([part_ent, ent["type"]]))
            ent_partial_offset_set.add(str([ent["tok_span"][0], ent["type"]]))

            ent_exact_text_set.add(str([ent["text"], ent["type"]]))
            ent_exact_offset_set.add(str([ent["type"]] + ent["tok_span"]))
            if sent_w_disc:
                ent_exact_offset_on_sents_w_disc_set.add(str([ent["type"]] + ent["tok_span"]))
                ent_exact_text_on_sents_w_disc_set.add(str([ent["text"], ent["type"]]))

            if len(ent["tok_span"]) > 2:
                disc_ent_exact_offset_set.add(str([ent["type"]] + ent["tok_span"]))
                disc_ent_exact_text_set.add(str([ent["text"], ent["type"]]))

        return {
            "ent_partial_text": ent_partial_text_set,
            "ent_partial_offset": ent_partial_offset_set,
            "ent_exact_text": ent_exact_text_set,
            "ent_exact_offset": ent_exact_offset_set,
            "disc_ent_exact_offset": disc_ent_exact_offset_set,
            "disc_ent_exact_text": disc_ent_exact_text_set,
            "ent_exact_offset_on_sents_w_disc": ent_exact_offset_on_sents_w_disc_set,
            "ent_exact_text_on_sents_w_disc": ent_exact_text_on_sents_w_disc_set,
        }
    
    @staticmethod
    def get_mark_sets_rel(rel_list):
        rel_partial_text_set, \
        rel_partial_offset_set, \
        rel_exact_text_set, \
        rel_exact_offset_set = set(), set(), set(), set()

        for rel in rel_list:
            part_subj = MetricsCalculator.get_partial_ent(rel["subject"])
            part_obj = MetricsCalculator.get_partial_ent(rel["subject"])
            rel_partial_text_set.add(str([part_subj, rel["predicate"], part_obj]))
            rel_exact_text_set.add(str([rel["subject"], rel["predicate"], rel["object"]]))
            rel_partial_offset_set.add(str([rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]]))
            rel_exact_offset_set.add(str(rel["subj_tok_span"] + [rel["predicate"]] + rel["obj_tok_span"]))
        return {
            "rel_partial_text": rel_partial_text_set,
            "rel_partial_offset": rel_partial_offset_set,
            "rel_exact_text": rel_exact_text_set,
            "rel_exact_offset": rel_exact_offset_set,
        }

    @staticmethod
    def cal_cpg(pred_set, gold_set, cpg):
        '''
        cpg is a list: [correct_num, pred_num, gold_num]
        '''
        correct_num = 0
        for mark_str in pred_set:
            if mark_str in gold_set:
                correct_num += 1
            # else:
            #     raise Exception("debug")

        cpg[0] += correct_num
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

        # if len(pred_set) != len(gold_set):
        #     raise Exception("debug")

    @staticmethod
    def get_mark_sets4disc_ent_analysis(ent_list):
        keys = {"no_overlap", "left_overlap", "right_overlap", "inner_overlap", "multi_overlap",
                "span_len: 3", "span_len: 4", "span_len: 5", "span_len: 6", "span_len: 7", "span_len: 8", "span_len: 9+",
                "interval_len: 4", "interval_len: 3", "interval_len: 2", "interval_len: 1", "interval_len: 5", "interval_len: 6", "interval_len: 7+"}
        mark_set_dict = {k: set() for k in keys}

        tok_id2occur_num = {}
        for ent in ent_list:
            tok_span = ent["tok_span"]
            # if len(tok_span) == 2:
            #     continue
            tok_ids = utils.spans2ids(tok_span)
            for tok_idx in tok_ids:
                tok_id2occur_num[tok_idx] = tok_id2occur_num.get(tok_idx, 0) + 1

        for ent in ent_list:
            mark = str([ent["type"]] + ent["tok_span"])
            tok_span = ent["tok_span"]
            if len(tok_span) == 2:
                continue

            # length analysis
            span_len = tok_span[-1] - tok_span[0]
            interval_len = 0
            for i in range(1, len(tok_span), 2):
                if i == len(tok_span) - 1:
                    break
                interval_len += (tok_span[i + 1] - tok_span[i])
            span_len_str = "9+" if span_len >= 9 else str(span_len)
            interval_len_str = "7+" if interval_len >= 7 else str(interval_len)
            mark_set_dict["span_len: {}".format(span_len_str)].add(mark)
            mark_set_dict["interval_len: {}".format(interval_len_str)].add(mark)

            # overlap analysis
            tok_ids = utils.spans2ids(tok_span)
            left_overlap, right_overlap, inner_overlap = 0, 0, 0
            for idx, tok_pos in enumerate(tok_ids):
                if tok_id2occur_num[tok_pos] > 1:
                    if idx == 0:
                        left_overlap = 1
                    elif idx == len(tok_ids) - 1:
                        right_overlap = 1
                    elif tok_id2occur_num[tok_ids[0]] == 1 and tok_id2occur_num[tok_ids[-1]] == 1:
                        inner_overlap = 1
            # for idx in range(0, len(tok_span), 2):
            #     sp = [tok_span[idx], tok_span[idx + 1]]
            #     tok_ids = utils.spans2ids(sp)
            #     if any(tok_id2occur_num[tok_pos] > 1 for tok_pos in tok_ids):
            #         if idx == 0:
            #             left_overlap = 1
            #         elif idx == len(tok_span) - 2:
            #             right_overlap = 1
            #         else:
            #             inner_overlap = 1

            if left_overlap + right_overlap + inner_overlap > 1:
                mark_set_dict["multi_overlap"].add(mark)
            else:
                if left_overlap == 1:
                    mark_set_dict["left_overlap"].add(mark)
                elif right_overlap == 1:
                    mark_set_dict["right_overlap"].add(mark)
                elif inner_overlap == 1:
                    mark_set_dict["inner_overlap"].add(mark)
                else:
                    mark_set_dict["no_overlap"].add(mark)

        return mark_set_dict

    count = 0
    @staticmethod
    def cal_cpg4disc_ent_add_analysis(pred_ent_list, gold_ent_list, cpg_dict):
        '''
        keys = {"no_overlap", "left_overlap", "right_overlap", "inner_overlap", "multi_overlap",
                "span_len: 3", "span_len: 4", "span_len: 5", "span_len: 6", "span_len: 7", "span_len: 8", "span_len: 9+"
                "interval_len: 4", "interval_len: 3", "interval_len: 2", "interval_len: 1", "interval_len: 5", "interval_len: 6", "interval_len: 7+"}
        '''
        pred_set_dict = MetricsCalculator.get_mark_sets4disc_ent_analysis(pred_ent_list)
        gold_set_dict = MetricsCalculator.get_mark_sets4disc_ent_analysis(gold_ent_list)

        for key in cpg_dict.keys():
            pred_set, gold_set = pred_set_dict[key], gold_set_dict[key]
            # if len(gold_set) > 0 and key == "span_len: 3":
            #     MetricsCalculator.count += len(gold_set)
            #     print(MetricsCalculator.count)
            MetricsCalculator.cal_cpg(pred_set, gold_set, cpg_dict[key])

    @staticmethod
    def cal_ent_cpg(pred_ent_list, gold_ent_list, ent_cpg_dict, sent_w_disc=False):
        '''
        ent_cpg_dict = {
            "ent_partial_text": [0, 0, 0],
            "ent_partial_offset": [0, 0, 0],
            "ent_exact_text": [0, 0, 0],
            "ent_exact_offset": [0, 0, 0],
        }
        if compute disc:
        ent_cpg_dict["disc_ent_exact_offset"] = [0, 0, 0]
        ent_cpg_dict["disc_ent_exact_text"] = [0, 0, 0]
        ent_cpg_dict["ent_exact_offset_on_sents_w_disc"] = [0, 0, 0]
        ent_cpg_dict["ent_exact_text_on_sents_w_disc"] = [0, 0, 0]
        '''
        pred_set_dict = MetricsCalculator.get_mark_sets_ent(pred_ent_list, sent_w_disc)
        gold_set_dict = MetricsCalculator.get_mark_sets_ent(gold_ent_list, sent_w_disc)
        for key in ent_cpg_dict.keys():
            pred_set, gold_set = pred_set_dict[key], gold_set_dict[key]
            MetricsCalculator.cal_cpg(pred_set, gold_set, ent_cpg_dict[key])

    @staticmethod
    def cal_rel_cpg(pred_rel_list, gold_rel_list, re_cpg_dict):
        '''
        re_cpg_dict = {
            "rel_partial_text": [0, 0, 0],
            "rel_partial_offset": [0, 0, 0],
            "rel_exact_text": [0, 0, 0],
            "rel_exact_offset": [0, 0, 0],
        }
        '''
        pred_set_dict = MetricsCalculator.get_mark_sets_rel(pred_rel_list)
        gold_set_dict = MetricsCalculator.get_mark_sets_rel(gold_rel_list)
        for key in re_cpg_dict.keys():
            pred_set, gold_set = pred_set_dict[key], gold_set_dict[key]
            MetricsCalculator.cal_cpg(pred_set, gold_set, re_cpg_dict[key])

    @staticmethod
    def cal_ee_cpg(pred_event_list, gold_event_list, ee_cpg_dict):
        '''
        ee_cpg_dict = {
            "trigger_iden": [0, 0, 0],
            "trigger_class": [0, 0, 0],
            "arg_soft_iden": [0, 0, 0],
            "arg_soft_class": [0, 0, 0],
            "arg_hard_iden": [0, 0, 0],
            "arg_hard_class": [0, 0, 0],
            "arg_link_iden": [0, 0, 0],
            "arg_link_class": [0, 0, 0],
        }
        '''
        pred_set_dict = MetricsCalculator.get_mark_sets_ee(pred_event_list)
        gold_set_dict = MetricsCalculator.get_mark_sets_ee(gold_event_list)
        for key in ee_cpg_dict.keys():
            pred_set, gold_set = pred_set_dict[key], gold_set_dict[key]
            MetricsCalculator.cal_cpg(pred_set, gold_set, ee_cpg_dict[key])

    @staticmethod
    def get_ee_cpg_dict(pred_sample_list, golden_sample_list):
        ee_cpg_dict = {
            "trigger_iden": [0, 0, 0],
            "trigger_class": [0, 0, 0],
            "arg_soft_iden": [0, 0, 0],
            "arg_soft_class": [0, 0, 0],
            "arg_hard_iden": [0, 0, 0],
            "arg_hard_class": [0, 0, 0],
            "arg_link_iden": [0, 0, 0],
            "arg_link_class": [0, 0, 0],
        }
        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_event_list = pred_sample["event_list"]
            gold_event_list = gold_sample["event_list"]
            # try:
            MetricsCalculator.cal_ee_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
            # except Exception as e:
            #     print("event error!")
        return ee_cpg_dict

    def get_rel_cpg_dict(self, pred_sample_list, golden_sample_list):
        re_cpg_dict = {
            "rel_partial_text": [0, 0, 0],
            "rel_partial_offset": [0, 0, 0],
            "rel_exact_text": [0, 0, 0],
            "rel_exact_offset": [0, 0, 0],
        }
        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_rel_list = pred_sample["relation_list"]
            gold_rel_list = gold_sample["relation_list"]
            # try:
            self.cal_rel_cpg(pred_rel_list, gold_rel_list, re_cpg_dict)
            # except Exception:
            #     pass
        return re_cpg_dict

    @staticmethod
    def get_ent_cpg_dict(pred_sample_list, golden_sample_list):
        ent_cpg_dict = {
            "ent_partial_text": [0, 0, 0],
            "ent_partial_offset": [0, 0, 0],
            "ent_exact_text": [0, 0, 0],
            "ent_exact_offset": [0, 0, 0],
        }

        if any(len(ent["char_span"]) > 2 for sample in golden_sample_list for ent in sample["entity_list"]):
            ent_cpg_dict["disc_ent_exact_offset"] = [0, 0, 0]
            ent_cpg_dict["disc_ent_exact_text"] = [0, 0, 0]
            ent_cpg_dict["ent_exact_offset_on_sents_w_disc"] = [0, 0, 0]
            ent_cpg_dict["ent_exact_text_on_sents_w_disc"] = [0, 0, 0]

        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_ent_list = pred_sample["entity_list"]
            gold_ent_list = gold_sample["entity_list"]
            sent_w_disc = any(len(ent["tok_span"]) > 2 for ent in gold_ent_list)
            # try:
            MetricsCalculator.cal_ent_cpg(pred_ent_list, gold_ent_list, ent_cpg_dict, sent_w_disc)
            # except Exception:
            #     print("1")

        return ent_cpg_dict

    @staticmethod
    def do_additonal_analysis4disc_ent(pred_sample_list, golden_sample_list):
        keys = {"no_overlap", "left_overlap", "right_overlap", "inner_overlap", "multi_overlap", 
                "span_len: 3", "span_len: 4", "span_len: 5", "span_len: 6", "span_len: 7", "span_len: 8", "span_len: 9+",
                "interval_len: 4", "interval_len: 3", "interval_len: 2", "interval_len: 1", "interval_len: 5", "interval_len: 6", "interval_len: 7+"}
        cpg_dict = {k: [0, 0, 0] for k in keys}

        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_ent_list = pred_sample["entity_list"]
            gold_ent_list = gold_sample["entity_list"]
            MetricsCalculator.cal_cpg4disc_ent_add_analysis(pred_ent_list, gold_ent_list, cpg_dict)

        prf_dict = {}
        statistics = {}
        for k, cpg in cpg_dict.items():
            statistics[k] = cpg[2]
            prf_dict[k] = MetricsCalculator.get_prf_scores(*cpg)
        return prf_dict, statistics

    def get_ioe_score_dict(self, pred_sample_list, golden_sample_list):
        from InfExtraction.modules.ancient_eval4oie import OIEMetrics
        auc, prfc, _ = OIEMetrics.compare(pred_sample_list,
                                          golden_sample_list,
                                          OIEMetrics.binary_linient_tuple_match)

        return {
            "auc": auc,
            "precision": prfc[0],
            "recall": prfc[1],
            "f1": prfc[2],
            "confidence_threshold": prfc[3],
        }

    @staticmethod
    def get_prf_scores(correct_num, pred_num, gold_num):
        '''
        get precision, recall, and F1 score
        :param correct_num:
        :param pred_num:
        :param gold_num:
        :return:
        '''
        if correct_num == pred_num == gold_num == 0:
            return 1.2333, 1.2333, 1.2333  # highlight this info by illegal outputs instead of outputting 0.

        minimum = 1e-20
        precision = correct_num / (pred_num + minimum)
        recall = correct_num / (gold_num + minimum)
        f1 = 2 * precision * recall / (precision + recall + minimum)
        return round(precision, 5), round(recall, 5), round(f1, 5)

    def score(self, pred_data, golden_data, data_filename=""):
        if data_filename != "":
            data_filename += "_"

        assert len(golden_data) > 0
        golden_sample = golden_data[0]

        total_cpg_dict = {}
        if "entity_list" in golden_sample:
            cpg_dict = self.get_ent_cpg_dict(pred_data, golden_data)
            total_cpg_dict = {**cpg_dict, **total_cpg_dict}

        if "relation_list" in golden_sample:
            cpg_dict = self.get_rel_cpg_dict(pred_data, golden_data)
            total_cpg_dict = {**cpg_dict, **total_cpg_dict}

        if "event_list" in golden_sample:
            cpg_dict = self.get_ee_cpg_dict(pred_data, golden_data)
            total_cpg_dict = {**cpg_dict, **total_cpg_dict}

        # if "open_spo_list" in golden_sample:
        #     cpg_dict = self.get_oie_scores

        score_dict = {}
        for sc_pattern, cpg in total_cpg_dict.items():
            prf = self.get_prf_scores(*cpg)
            for idx, sct in enumerate(["prec", "rec", "f1"]):
                score_dict["{}{}_{}".format(data_filename, sc_pattern, sct)] = round(prf[idx], 5)

        if "open_spo_list" in golden_sample:
            oie_score_dict = self.get_ioe_score_dict(golden_data, golden_data)
            for sct, val in oie_score_dict.items():
                score_dict["{}{}".format(data_filename, sct)] = round(val, 5)
        return score_dict
