import difflib
import copy


class OIEMetrics:

    @staticmethod
    def trans(spo):
        new_spo = {}
        for role in spo:
            if role["type"] != "object":
                new_spo[role["type"]] = role
            else:
                if "object" not in new_spo:
                    new_spo["object"] = []
                new_spo["object"].append(role)
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
        correct_num = 0
        gold_num = 0
        pred_num = 0
        new_gold_data = copy.deepcopy(gold_data)

        for idx in range(len(pred_data)):
            gold_num += len(new_gold_data[idx]["open_spo_list"])
            pred_num += len(pred_data[idx]["open_spo_list"])
            for predicted_ex in pred_data[idx]["open_spo_list"]:
                ex_score = 0
                hit_idx = None
                for idx, gold_ex in enumerate(
                        new_gold_data[idx]["open_spo_list"]):
                    match_score = OIEMetrics.match(OIEMetrics.trans(predicted_ex), OIEMetrics.trans(gold_ex))
                    if match_score > ex_score:
                        ex_score = match_score
                        hit_idx = idx
                if ex_score > threshold:
                    correct_num += 1
                    del new_gold_data[idx][hit_idx]

        return correct_num, pred_num, gold_num
