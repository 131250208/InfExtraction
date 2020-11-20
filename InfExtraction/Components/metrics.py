import torch

class Metrics:
    def __init__(self, tagger):
        super().__init__()
        self.tagger = tagger

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        This function is a loss function for multi-label learning
        ref: https://kexue.fm/archives/7359

        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = y_pred.float()
        y_true = y_true.float()

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def loss_func(self, y_pred, y_true):
        return self._multilabel_categorical_crossentropy(y_pred, y_true)

    def get_tag_accuracy(self, pred, truth):
        '''
        the tag accuracy in a batch
        a predicted tag sequence (matrix) is correct if and only if it is totally congruent with the ground truth
        '''
        #     # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        #     pred_id = torch.argmax(pred, dim = -1).int()

        # (batch_size, ..., seq_len) -> (batch_size, -1)
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        tag_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        tag_acc = torch.mean(tag_acc_, axis=0)

        return tag_acc

    def get_cpg(self, gold_sample_list,
                      tok2char_span_list,
                      pred_tag_matrix_batch):
        '''
        cpg: correct number, precision number, gold number
        :param gold_sample_list: golden sample
        :param tok2char_span_list: a list of maps from token index to character level spans,
                                    each map corresponds to a predicted matrix in the batch
        :param pred_tag_matrix_batch: a batch of predicted tag matrix
        :return:
        '''
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        for ind in range(len(gold_sample_list)):
            gold_sample = gold_sample_list[ind]
            text = gold_sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_tag_matrix = pred_tag_matrix_batch[ind]
            pred_event_list = self.tagger.decode(text, pred_tag_matrix, tok2char_span)
            gold_event_list = gold_sample["entity_list"]

            self.cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict)

        return ee_cpg_dict

    def get_scores(self, correct_num, pred_num, gold_num):
        '''
        get precision, recall, and F1 score
        :param correct_num:
        :param pred_num:
        :param gold_num:
        :return:
        '''
        minimini = 1e-12
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1

    def _cal_cpg(self, pred_set, gold_set, cpg):
        '''
        cpg is a list: [correct_num, pred_num, gold_num]
        '''
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

    def cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
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
        pred_arg_class_set = self.get_mark_sets(pred_event_list)

        gold_trigger_iden_set, \
        gold_trigger_class_set, \
        gold_arg_iden_set, \
        gold_arg_class_set = self.get_mark_sets(gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set, ee_cpg_dict["trigger_iden_cpg"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set, ee_cpg_dict["trigger_class_cpg"])
        self._cal_cpg(pred_arg_iden_set, gold_arg_iden_set, ee_cpg_dict["arg_iden_cpg"])
        self._cal_cpg(pred_arg_class_set, gold_arg_class_set, ee_cpg_dict["arg_class_cpg"])

    def get_mark_sets(self, event_list):
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