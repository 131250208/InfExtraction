import torch
import re


class MetricsCalculator:
    def __init__(self, task_type, match_pattern=None, use_ghm=False):
        self.task_type = task_type  # for scoring
        self.match_pattern = match_pattern  # for scoring of relation extraction
        if task_type == "re":
            assert self.match_pattern is not None
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
            hits = torch.sum((gradient_norm <= bar)) - count_hits
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

        y_pred: (batch_size_train, shaking_seq_len, type_size)
        y_true: (batch_size_train, shaking_seq_len, type_size)
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

    def _get_mark_sets_ee(self, event_list):
        trigger_iden_set, trigger_class_set = set(), set()
        arg_hard_iden_set, arg_hard_class_set = set(), set()  # consider trigger offset
        arg_soft_iden_set, arg_soft_class_set = set(), set()  # do not consider trigger offset
        arg_link_iden_set, arg_link_class_set = set(), set()  # for trigger-free metrics

        for event in event_list:
            # trigger-based metrics
            trigger_offset = None
            if "trigger" in event:
                event_type = event["trigger_type"]
                trigger_offset = event["trigger_tok_span"]
                trigger_iden_set.add("{}\u2E80{}".format(*trigger_offset))
                trigger_class_set.add("{}\u2E80{}\u2E80{}".format(event_type, *trigger_offset))

            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                event_type = arg["event_type"]

                arg_soft_iden_set.add(
                    "{}\u2E80{}\u2E80{}".format(event_type, *argument_offset))

                arg_soft_class_set.add("{}\u2E80{}\u2E80{}\u2E80{}".format(event_type,
                                                                           *argument_offset,
                                                                           argument_role))
                if "trigger" in event:
                    assert trigger_offset is not None
                    arg_hard_iden_set.add(
                        "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type,
                                                                    *argument_offset,
                                                                    *trigger_offset,
                                                                    ))

                    arg_hard_class_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type,
                                                                                               *argument_offset,
                                                                                               *trigger_offset,
                                                                                               argument_role))

            # cal trigger-free metrics
            arg_list = event["argument_list"]
            if "trigger" in event:
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

                    link_iden_mark = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(arg_i_event_type,
                                                                                         arg_j_event_type,
                                                                                         *arg_i_offset,
                                                                                         *arg_j_offset)
                    link_class_mark = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                        arg_i_event_type,
                        arg_j_event_type,
                        *arg_i_offset,
                        *arg_j_offset,
                        arg_i_role,
                        arg_j_role)
                    arg_link_iden_set.add(link_iden_mark)
                    arg_link_class_set.add(link_class_mark)

        return trigger_iden_set, \
               trigger_class_set, \
               arg_soft_iden_set, \
               arg_soft_class_set, \
               arg_hard_iden_set, \
               arg_hard_class_set, \
               arg_link_iden_set, \
               arg_link_class_set

    def _get_mark_sets_rel(self, rel_list, ent_list, match_pattern="only_head_text"):
        rel_set, ent_set = None, None

        if match_pattern == "only_head_index":
            rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
                 in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["tok_span"][0], ent["type"]) for ent in ent_list])

        elif match_pattern == "whole_span":
            rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1],
                                                                       rel["predicate"], rel["obj_tok_span"][0],
                                                                       rel["obj_tok_span"][1]) for rel in rel_list])
            ent_set = set(
                ["{}\u2E80{}\u2E80{}".format(ent["tok_span"][0], ent["tok_span"][1], ent["type"]) for ent in ent_list])

        elif match_pattern == "whole_text":
            rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"], ent["type"]) for ent in ent_list])

        elif match_pattern == "only_head_text":
            rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                       rel["object"].split(" ")[0]) for rel in rel_list])
            ent_set = set(["{}\u2E80{}".format(ent["text"].split(" ")[0], ent["type"]) for ent in ent_list])

        return rel_set, ent_set

    def _cal_cpg(self, pred_set, gold_set, cpg):
        '''
        cpg is a list: [correct_num, pred_num, gold_num]
        '''
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1

        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)
        # if len(pred_set) != len(gold_set):
        #     print("!")

    def _cal_rel_cpg(self, pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern):
        '''
        ere_cpg_dict = {
                "rel": [0, 0, 0], # correct num, precise num, golden num
                "ent": [0, 0, 0],
                }
        pattern: metric pattern
        '''
        # pred_rel_list = [rel for rel in pred_rel_list if rel["predicate"].split(":")[0] not in {"EE"}]
        # pred_ent_list = [ent for ent in pred_ent_list if ent["type"].split(":")[0] not in {"EXT", "EE"}]

        # filter extra entities
        ent_type_set = {ent["type"] for ent in gold_ent_list}
        ent_types2filter = {"REL:", "EE:"}
        if len(ent_type_set) == 1 and list(ent_type_set)[0] == "EXT:DEFAULT":
            pass
        else:
            ent_types2filter.add("EXT:")
        filter_pattern = "({})".format("|".join(ent_types2filter))
        gold_ent_list = [ent for ent in gold_ent_list if re.search(filter_pattern, ent["type"]) is None]

        gold_rel_set, gold_ent_set = self._get_mark_sets_rel(gold_rel_list, gold_ent_list, pattern)
        pred_rel_set, pred_ent_set = self._get_mark_sets_rel(pred_rel_list, pred_ent_list, pattern)

        self._cal_cpg(pred_rel_set, gold_rel_set, ere_cpg_dict["rel"])
        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent"])

    def _cal_ee_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
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
        pred_trigger_iden_set, \
        pred_trigger_class_set, \
        pred_arg_soft_iden_set, \
        pred_arg_soft_class_set, \
        pred_arg_hard_iden_set, \
        pred_arg_hard_class_set, \
        pred_arg_link_iden_set, \
        pred_arg_link_class_set = self._get_mark_sets_ee(pred_event_list)

        gold_trigger_iden_set, \
        gold_trigger_class_set, \
        gold_arg_soft_iden_set, \
        gold_arg_soft_class_set, \
        gold_arg_hard_iden_set, \
        gold_arg_hard_class_set, \
        gold_arg_link_iden_set, \
        gold_arg_link_class_set = self._get_mark_sets_ee(gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set, ee_cpg_dict["trigger_iden"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set, ee_cpg_dict["trigger_class"])
        self._cal_cpg(pred_arg_soft_iden_set, gold_arg_soft_iden_set, ee_cpg_dict["arg_soft_iden"])
        self._cal_cpg(pred_arg_soft_class_set, gold_arg_soft_class_set, ee_cpg_dict["arg_soft_class"])
        self._cal_cpg(pred_arg_hard_iden_set, gold_arg_hard_iden_set, ee_cpg_dict["arg_hard_iden"])
        self._cal_cpg(pred_arg_hard_class_set, gold_arg_hard_class_set, ee_cpg_dict["arg_hard_class"])
        self._cal_cpg(pred_arg_link_iden_set, gold_arg_link_iden_set, ee_cpg_dict["arg_link_iden"])
        self._cal_cpg(pred_arg_link_class_set, gold_arg_link_class_set, ee_cpg_dict["arg_link_class"])

    def get_ee_cpg_dict(self, pred_sample_list, golden_sample_list):
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
            self._cal_ee_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
        return ee_cpg_dict

    def get_rel_cpg_dict(self, pred_sample_list, golden_sample_list, match_pattern):
        ere_cpg_dict = {
            "rel": [0, 0, 0],
            "ent": [0, 0, 0],
        }
        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_rel_list = pred_sample["relation_list"]
            gold_rel_list = gold_sample["relation_list"]
            pred_ent_list = pred_sample["entity_list"]
            gold_ent_list = gold_sample["entity_list"]
            self._cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, match_pattern)
        return ere_cpg_dict

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        '''
        get precision, recall, and F1 score
        :param correct_num:
        :param pred_num:
        :param gold_num:
        :return:
        '''
        minimum = 1e-20
        precision = correct_num / (pred_num + minimum)
        recall = correct_num / (gold_num + minimum)
        f1 = 2 * precision * recall / (precision + recall + minimum)
        return precision, recall, f1

    def score(self, pred_data, golden_data, data_filename=""):
        if data_filename != "":
            data_filename += "_"

        total_cpg_dict = {}
        if "re" in self.task_type:
            assert self.match_pattern is not None
            cpg_dict = self.get_rel_cpg_dict(pred_data,
                                             golden_data,
                                             self.match_pattern)
            total_cpg_dict = {**cpg_dict, **total_cpg_dict}

        if "ee" in self.task_type:
            cpg_dict = self.get_ee_cpg_dict(pred_data, golden_data)
            total_cpg_dict = {**cpg_dict, **total_cpg_dict}

        score_dict = {}
        for key, cpg in total_cpg_dict.items():
            prf = self.get_prf_scores(*cpg)
            for idx, sct in enumerate(["prec", "recall", "f1"]):
                score_dict["{}{}_{}".format(data_filename, key, sct)] = round(prf[idx], 5)

        return score_dict
