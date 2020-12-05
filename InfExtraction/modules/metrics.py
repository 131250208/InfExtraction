import torch


class MetricsCalculator:
    def __init__(self, task_type, match_pattern=None, use_ghm=False):
        self.task_type = task_type # for scoring
        self.match_pattern = match_pattern # for scoring of relation extraction
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

    def _get_mark_sets_event(self, event_list):
        trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set = set(), set(), set(), set()
        for event in event_list:
            event_type = event["trigger_type"]
            trigger_offset = event["trigger_tok_span"]
            trigger_iden_set.add("{}\u2E80{}".format(trigger_offset[0], trigger_offset[1]))
            trigger_class_set.add("{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0], trigger_offset[1]))
            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                arg_iden_set.add(
                    "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0], trigger_offset[1],
                                                                argument_offset[0], argument_offset[1]))
                arg_class_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(event_type, trigger_offset[0],
                                                                                      trigger_offset[1],
                                                                                      argument_offset[0],
                                                                                      argument_offset[1],
                                                                                      argument_role))

        return trigger_iden_set, \
               trigger_class_set, \
               arg_iden_set, \
               arg_class_set

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

    def _cal_rel_cpg(self, pred_rel_list, pred_ent_list, gold_rel_list, gold_ent_list, ere_cpg_dict, pattern):
        '''
        ere_cpg_dict = {
                "rel_cpg": [0, 0, 0], # correct num, precise num, golden num
                "ent_cpg": [0, 0, 0],
                }
        pattern: metric pattern
        '''
        pred_rel_list = [rel for rel in pred_rel_list if rel["predicate"].split(":")[0] not in {"EE", "NER"}]
        pred_ent_list = [ent for ent in pred_ent_list if ent["type"].split(":")[0] not in {"EXT", "EE"}]
        gold_rel_list = [rel for rel in gold_rel_list if rel["predicate"].split(":")[0] not in {"EE", "NER"}]
        gold_ent_list = [ent for ent in gold_ent_list if ent["type"].split(":")[0] not in {"EXT", "EE"}]

        gold_rel_set, gold_ent_set = self._get_mark_sets_rel(gold_rel_list, gold_ent_list, pattern)
        pred_rel_set, pred_ent_set = self._get_mark_sets_rel(pred_rel_list, pred_ent_list, pattern)

        self._cal_cpg(pred_rel_set, gold_rel_set, ere_cpg_dict["rel_cpg"])
        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent_cpg"])

    def _cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
        '''
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        '''
        pred_trigger_iden_set, \
        pred_trigger_class_set, \
        pred_arg_iden_set, \
        pred_arg_class_set = self._get_mark_sets_event(pred_event_list)

        gold_trigger_iden_set, \
        gold_trigger_class_set, \
        gold_arg_iden_set, \
        gold_arg_class_set = self._get_mark_sets_event(gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set, ee_cpg_dict["trigger_iden_cpg"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set, ee_cpg_dict["trigger_class_cpg"])
        # if len(pred_arg_iden_set) != len(gold_arg_iden_set): # 解码算法引入的错误，可以作为下一个研究点，即群里讨论过的极端嵌套情况
        #     print("!")
        self._cal_cpg(pred_arg_iden_set, gold_arg_iden_set, ee_cpg_dict["arg_iden_cpg"])
        self._cal_cpg(pred_arg_class_set, gold_arg_class_set, ee_cpg_dict["arg_class_cpg"])

    def get_event_cpg_dict(self, pred_sample_list, golden_sample_list):
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        for idx, pred_sample in enumerate(pred_sample_list):
            gold_sample = golden_sample_list[idx]
            pred_event_list = pred_sample["event_list"]
            gold_event_list = gold_sample["event_list"]
            self._cal_event_cpg(pred_event_list, gold_event_list, ee_cpg_dict)
        return ee_cpg_dict

    def get_rel_cpg_dict(self, pred_sample_list, golden_sample_list, match_pattern):
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
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
    
    def score(self, pred_data, golden_data, data_type):
        score_dict = None
        if self.task_type == "re":
            assert self.match_pattern is not None
            total_cpg_dict = self.get_rel_cpg_dict(pred_data,
                                                 golden_data,
                                                 self.match_pattern)
            rel_prf = self.get_prf_scores(*total_cpg_dict["rel_cpg"])
            ent_prf = self.get_prf_scores(*total_cpg_dict["ent_cpg"])
            score_dict = {
                "{}_rel_prec".format(data_type): rel_prf[0],
                "{}_rel_recall".format(data_type): rel_prf[1],
                "{}_rel_f1".format(data_type): rel_prf[2],
                "{}_ent_prec".format(data_type): ent_prf[0],
                "{}_ent_recall".format(data_type): ent_prf[1],
                "{}_ent_f1".format(data_type): ent_prf[2]
            }

        elif self.task_type == "ee":
            total_cpg_dict = self.get_event_cpg_dict(pred_data, golden_data)
            trigger_iden_prf = self.get_prf_scores(*total_cpg_dict["trigger_iden_cpg"])
            trigger_class_prf = self.get_prf_scores(*total_cpg_dict["trigger_class_cpg"])
            arg_iden_prf = self.get_prf_scores(*total_cpg_dict["arg_iden_cpg"])
            arg_class_prf = self.get_prf_scores(*total_cpg_dict["arg_class_cpg"])

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

        return score_dict