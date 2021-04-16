import difflib
import copy


class OIEMetrics:

    @staticmethod
    def trans(spo):
        new_spo = {}
        for arg in spo:
            if arg["type"] != "object":
                new_spo[arg["type"]] = arg
            else:
                if "object" not in new_spo:
                    new_spo["object"] = []
                new_spo["object"].append(arg)
        return new_spo

    @staticmethod
    def match(predicted_ex, gold_ex):
        match_score = 0
        element_num = 1e-20

        for key in set(predicted_ex.keys()).union(set(gold_ex.keys())):
            if key != "object":
                element_num += 1
            else:
                predicted_obj_num = len(predicted_ex["object"]) if "object" in predicted_ex else 0
                gold_obj_num = len(
                    gold_ex["object"]) if "object" in gold_ex else 0
                element_num += max(predicted_obj_num, gold_obj_num)

        for tp in predicted_ex:
            if tp in gold_ex:
                if tp != "object":
                    match_score += difflib.SequenceMatcher(
                        None, predicted_ex[tp]["text"],
                        gold_ex[tp]["text"]).ratio()
                else:
                    min_object_num = min(len(predicted_ex["object"]), len(gold_ex["object"]))
                    for idx in range(min_object_num):
                        match_score += difflib.SequenceMatcher(
                            None, predicted_ex["object"][idx]["text"],
                            gold_ex["object"][idx]["text"]).ratio()

        return match_score / element_num

    @staticmethod
    def compare(pred_data, gold_data, threshold):
        # 读每个ins，计算每个pair的相似性，
        total_correct_num = 0
        total_gold_num = 0
        total_pred_num = 0

        for sample_idx, pred_sample in enumerate(pred_data):
            gold_sample = gold_data[sample_idx]
            pred_spo_list = pred_sample["open_spo_list"]
            gold_spo_list4debug = gold_sample["open_spo_list"]
            gold_spo_list = copy.deepcopy(gold_sample["open_spo_list"])

            pred_num = len(pred_spo_list)
            gold_num = len(gold_spo_list)

            total_gold_num += gold_num
            total_pred_num += pred_num

            correct_num = 0
            for predicted_ex in pred_spo_list:
                ex_score = 0
                hit_idx = None
                for spo_idx, gold_ex in enumerate(
                        gold_spo_list):
                    match_score = OIEMetrics.match(OIEMetrics.trans(predicted_ex), OIEMetrics.trans(gold_ex))
                    if match_score > ex_score:
                        ex_score = match_score
                        hit_idx = spo_idx
                if ex_score > threshold:
                    correct_num += 1
                    del gold_spo_list[hit_idx]
            total_correct_num += correct_num

            # if not (correct_num == pred_num == gold_num):
            #     print("!")
        return total_correct_num, total_pred_num, total_gold_num
