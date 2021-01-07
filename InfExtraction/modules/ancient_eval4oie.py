import numpy as np
from sklearn.metrics import auc
from copy import copy
from _collections import defaultdict
from pattern.en import lexeme, lemma

class Extraction:
    """
    Stores sentence, single predicate and corresponding arguments.
    """
    def __init__(self, pred, head_pred_index, sent, confidence, question_dist = '', index = -1):
        self.pred = pred
        self.head_pred_index = head_pred_index
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        self.is_mwp = False
        self.question_dist = question_dist
        self.index = index

    def add_arg(self, arg):
        self.args.append(arg)

class OIEMetrics:
    PREPS = ['above', 'across', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath',
             'beside', 'between', 'by', 'for', 'from', 'in', 'into', 'near', 'of', 'off', 'on', 'to', 'toward', 'under',
             'upon', 'with', 'within']

    @staticmethod
    def trans_2extra_obj(data):
        text2spo = dict()
        for sample in data:
            text = sample["text"]
            for spo in sample["open_spo_list"]:
                extr_spo = Extraction(pred=spo["predicate"]["complete"],
                                      head_pred_index=None,
                                      sent=text,
                                      confidence=1.)
                extr_spo.add_arg(spo["subject"]["text"])
                object_ = " ".join([spo["object"]["text"]] + [arg["text"] for arg in spo["other_args"]])
                extr_spo.add_arg(object_)
                for arg in spo["other_args"]:
                    extr_spo.add_arg(arg["text"])
                if text not in text2spo:
                    text2spo[text] = []
                text2spo[text].append(extr_spo)
        return text2spo

    @staticmethod
    def linient_tuple_match(ref, ex):
        precision = [0, 0]  # 0 out of 0 predicted words match
        recall = [0, 0]  # 0 out of 0 reference words match

        # If, for each part, any word is the same as a reference word, then it's a match.

        def my_lemma(word):
            if word[-2:] in {"'s", "ly", "'t"}:
                word = word[:-2]
            word = lemma(word)
            return word

        predicted_words = [my_lemma(w) for w in ex.pred.split()]
        gold_words = [my_lemma(w) for w in ref.pred.split()]
        precision[1] += len(predicted_words)
        recall[1] += len(gold_words)

        # matching_words = sum(1 for w in predicted_words if w in gold_words)
        matching_words = 0
        for w in gold_words:
            if w in predicted_words:
                matching_words += 1
                predicted_words.remove(w)

        # # matching 'be' with its different forms
        # forms_of_be = lexeme("be")
        # if "be" in predicted_words:
        #     for form in forms_of_be:
        #         if form in gold_words:
        #             matching_words += 1
        #             predicted_words.remove("be")
        #             break

        if matching_words == 0:
            return [0, 0]  # t <-> gt is not a match

        precision[0] += matching_words
        recall[0] += matching_words

        for i in range(len(ref.args)):
            gold_words = [my_lemma(w) for w in ref.args[i].split()]
            recall[1] += len(gold_words)
            if len(ex.args) <= i:
                if i < 2:
                    return [0, 0]  # changed
                else:
                    continue
            predicted_words = [my_lemma(w) for w in ex.args[i].split()]
            precision[1] += len(predicted_words)
            matching_words = 0
            for w in gold_words:
                if w in predicted_words:
                    matching_words += 1
                    predicted_words.remove(w)

            precision[0] += matching_words
            # Currently this slightly penalises systems when the reference
            # reformulates the sentence words, because the reformulation doesn't
            # match the predicted word. It's a one-wrong-word penalty to precision,
            # to all systems that correctly extracted the reformulated word.
            recall[0] += matching_words

        if (precision[1] == 0):
            prec = 0
        else:
            prec = 1.0 * precision[0] / precision[1]
        if (recall[1] == 0):
            rec = 0
        else:
            rec = 1.0 * recall[0] / recall[1]

        return [prec, rec]

    @staticmethod
    def binary_linient_tuple_match(ref, ex):
        if len(ref.args) >= 2:
            r = copy(ref)
            r.args = [ref.args[0], ' '.join(ref.args[1:])]
        else:
            r = ref
        if len(ex.args) >= 2:
            e = copy(ex)
            e.args = [ex.args[0], ' '.join(ex.args[1:])]
        else:
            e = ex

        stright_match = OIEMetrics.linient_tuple_match(r, e)

        said_type_reln = lexeme("say") + lexeme("tell") + lexeme("add")
        said_type_sentence = False
        for said_verb in said_type_reln:
            if said_verb in ref.pred:
                said_type_sentence = True
                break
        if not said_type_sentence:
            return stright_match
        else:
            if len(ex.args) >= 2:
                e = copy(ex)
                e.args = [' '.join(ex.args[1:]), ex.args[0]]
            else:
                e = ex
            reverse_match = OIEMetrics.linient_tuple_match(r, e)

            return max(stright_match, reverse_match)

    @staticmethod
    def compare(pred_data, gold_data, matchingFunc, binary=False, strategy='sm'):
        pred_data = OIEMetrics.trans_2extra_obj(pred_data)
        gold_data = OIEMetrics.trans_2extra_obj(gold_data)

        if binary:
            pred_data = OIEMetrics.binarize(pred_data)
            gold_data = OIEMetrics.binarize(gold_data)
        # taking all distinct values of confidences as thresholds
        confidence_thresholds = set()
        for sent in pred_data:
            for predicted_ex in pred_data[sent]:
                confidence_thresholds.add(predicted_ex.confidence)

        confidence_thresholds = sorted(list(confidence_thresholds))
        num_conf = len(confidence_thresholds)

        p = np.zeros(num_conf)
        pl = np.zeros(num_conf)
        r = np.zeros(num_conf)
        rl = np.zeros(num_conf)

        for sent, goldExtractions in gold_data.items():
            if sent in pred_data:
                predictedExtractions = pred_data[sent]
            else:
                predictedExtractions = []
                # continue # Uncomment if you want to ignore gold_data sentences with no predictions

            scores = [[None for _ in predictedExtractions] for __ in goldExtractions]

            for i, goldEx in enumerate(goldExtractions):
                for j, predictedEx in enumerate(predictedExtractions):
                    score = matchingFunc(goldEx, predictedEx)
                    scores[i][j] = score

            # OPTIMISED GLOBAL MATCH
            sent_confidences = [extraction.confidence for extraction in predictedExtractions]
            sent_confidences.sort()
            prev_c = 0
            for conf in sent_confidences:
                c = confidence_thresholds.index(conf)
                ext_indices = []
                for ext_indx, extraction in enumerate(predictedExtractions):
                    if extraction.confidence >= conf:
                        ext_indices.append(ext_indx)

                # ksk mod
                if strategy == 'sm':
                    recall_numerator = 0
                    for i, row in enumerate(scores):
                        max_recall_row = max([row[ext_indx][1] for ext_indx in ext_indices], default=0)
                        recall_numerator += max_recall_row

                precision_numerator = 0
                # for ext_indx in ext_indices:
                #     max_precision_col = max([scores[row_indx][ext_indx][0] for row_indx in range(len(scores)) if scores[row_indx][ext_indx] != (0,0)], default = 1)
                #     precision_numerator += max_precision_col
                selected_rows = []
                selected_cols = []
                num_precision_matches = min(len(scores), len(ext_indices))
                for t in range(num_precision_matches):
                    matched_row = -1
                    matched_col = -1
                    matched_precision = -1  # initialised to <0 so that it updates whenever precision is 0 as well
                    for i in range(len(scores)):
                        if i in selected_rows:
                            continue
                        for ext_indx in ext_indices:
                            if ext_indx in selected_cols:
                                continue
                            if scores[i][ext_indx][0] > matched_precision:
                                matched_precision = scores[i][ext_indx][0]
                                matched_row = i
                                matched_col = ext_indx

                    if matched_col == -1 or matched_row == -1:
                        raise Exception("error in CaRB, matched row/col is -1")

                    selected_rows.append(matched_row)
                    selected_cols.append(matched_col)
                    precision_numerator += scores[matched_row][matched_col][0]

                # ksk mod
                if strategy == 'ss':
                    recall_numerator = 0
                    selected_rows = []
                    selected_cols = []
                    num_recall_matches = min(len(scores), len(ext_indices))
                    for t in range(num_recall_matches):
                        matched_row = -1
                        matched_col = -1
                        matched_recall = -1  # initialised to <0 so that it updates whenever recall is 0 as well
                        for i in range(len(scores)):
                            if i in selected_rows:
                                continue
                            for ext_indx in ext_indices:
                                if ext_indx in selected_cols:
                                    continue
                                if scores[i][ext_indx][1] > matched_recall:
                                    matched_recall = scores[i][ext_indx][1]
                                    matched_row = i
                                    matched_col = ext_indx

                        if matched_col == -1 or matched_row == -1:
                            raise Exception("error in CaRB, matched row/col is -1")

                        selected_rows.append(matched_row)
                        selected_cols.append(matched_col)
                        recall_numerator += scores[matched_row][matched_col][1]

                p[prev_c:c + 1] += precision_numerator
                pl[prev_c:c + 1] += len(ext_indices)
                # pl[prev_c:c+1] += num_precision_matches
                r[prev_c:c + 1] += recall_numerator
                rl[prev_c:c + 1] += len(scores)

                prev_c = c + 1

            # for indices beyond the maximum sentence confidence, len(scores) has to be added to the denominator of recall
            rl[prev_c:] += len(scores)

        prec_scores = [a / b if b > 0 else 1 for a, b in zip(p, pl)]
        rec_scores = [a / b if b > 0 else 0 for a, b in zip(r, rl)]

        f1s = [OIEMetrics.f1(p, r) for p, r in zip(prec_scores, rec_scores)]

        try:
            optimal_idx = np.nanargmax(f1s)
            optimal = (
            np.round(prec_scores[optimal_idx], 4), np.round(rec_scores[optimal_idx], 4), np.round(f1s[optimal_idx], 4),
            confidence_thresholds[optimal_idx])
            zero_conf_point = (
            np.round(prec_scores[0], 4), np.round(rec_scores[0], 4), np.round(f1s[0], 4), confidence_thresholds[0])
        except ValueError:
            # When there is no prediction
            optimal = (0, 0, 0, 0)
            zero_conf_point = (0, 0, 0, 0)

        # In order to calculate auc, we need to add the point corresponding to precision=1 , recall=0 to the PR-curve
        temp_rec_scores = rec_scores.copy()
        temp_prec_scores = prec_scores.copy()
        temp_rec_scores.append(0)
        temp_prec_scores.append(1)
        # print("AUC: {}\t Optimal (precision, recall, F1): {}".format( np.round(auc(temp_rec_scores, temp_prec_scores),3), np.round(optimal,3) ))

        # with open(output_fn, 'w') as fout:
        #     fout.write('{0}\t{1}\t{2}\n'.format("Precision", "Recall", "Confidence"))
        #     for cur_p, cur_r, cur_conf in sorted(zip(prec_scores, rec_scores, confidence_thresholds),
        #                                          key=lambda cur: cur[1]):
        #         fout.write('{0}\t{1}\t{2}\n'.format(cur_p, cur_r, cur_conf))

        if len(f1s) > 0:
            return np.round(auc(temp_rec_scores, temp_prec_scores),
                            4), optimal, zero_conf_point
        else:
            # When there is no prediction
            return 0, (0, 0, 0, 0), (0, 0, 0, 0)

    @staticmethod
    def binarize(extrs):
        res = defaultdict(lambda: [])
        for sent, extr in extrs.items():
            for ex in extr:
                # Add (a1, r, a2)
                temp = copy(ex)
                temp.args = ex.args[:2]
                res[sent].append(temp)

                if len(ex.args) <= 2:
                    continue

                # Add (a1, r a2 , a3 ...)
                for arg in ex.args[2:]:
                    temp.args = [ex.args[0]]
                    temp.pred = ex.pred + ' ' + ex.args[1]
                    words = arg.split()

                    # Add preposition of arg to rel
                    if words[0].lower() in OIEMetrics.PREPS:
                        temp.pred += ' ' + words[0]
                        words = words[1:]
                    temp.args.append(' '.join(words))
                    res[sent].append(temp)

        return res

    @staticmethod
    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec + 1e-20)
