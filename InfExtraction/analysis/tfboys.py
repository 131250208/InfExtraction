from InfExtraction.work_flows import format_conv
from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import load_data
from InfExtraction.modules.preprocess import Preprocessor
from pprint import pprint
import pandas as pd


def convert_casee2normal_format(data):
    new_data = []
    for sample in data:
        new_sample = {
            "id": sample["id"],
            "text": sample["content"],
            "event_list": [],
        }
        for event in sample["events"]:
            new_event = {
                "event_type": event["type"],
                "trigger": event["trigger"]["word"],
                "trigger_tok_span": event["trigger"]["span"],
                "argument_list": [],
            }

            for role, args in event["args"].items():
                for arg in args:
                    new_event["argument_list"].append({
                        "text": arg["word"],
                        "tok_span": arg["span"],
                        "type": role,
                    })
            new_sample["event_list"].append(new_event)
        new_data.append(new_sample)
    return new_data


if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>> cal scores >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    metrics_key = ["trigger_iden", "trigger_class",
                   "arg_hard_iden", "arg_hard_class",
                   "arg_soft_class", "arg_class_most_similar_event",
                   "n_otm_arg_hard_class", "s_otm_arg_hard_class",
                   "n_otm_arg_class_most_similar_event", "s_otm_arg_class_most_similar_event"]
    main_res_dict = {}

    def fill_scores(scd, bsl):
        for k in metrics_key:
            scores = [str(round(scd["{}_{}".format(k, postfix)] * 100, 1)) for postfix in ["prec", "rec", "f1"]]
            score_str = "/".join(scores)
            main_res_dict.setdefault(k, {})[bsl] = score_str

    # # dbrnn, dmcnn, dygie, oneie
    # for baseline in ["dmcnn", "dbrnn", "dygiepp", "oneie"]:
    #     baseline_pred_data, baseline_gold_data = format_conv.convert_tfboys_baselines2normal_format("fewfc", baseline)
    #     score_dict = MetricsCalculator.score(baseline_pred_data, baseline_gold_data)
    #     fill_scores(score_dict, baseline)

    # casee
    casee_pred_data = load_data("../../data/tfboys_baselines/fewfc/casee/pred.json")
    casee_pred_data = convert_casee2normal_format(casee_pred_data)
    casee_gold_data = load_data("../../data/ori_data/few_fc_bk/test.json")
    casee_gold_data = convert_casee2normal_format(casee_gold_data)
    score_dict = MetricsCalculator.score(casee_pred_data, casee_gold_data)
    fill_scores(score_dict, "casee")

    # tbee BASE# tbee_model_path = "../../data/res_data/few_fc/re+tbee+RAIN+TRAIN/2uq6f74s/model_state_dict_22_71.846/test_data.json"
    tbee_model_path = "../../data/res_data/few_fc/re+ee+RAIN+TRAIN/zyuy3ra0/model_state_dict_46_66.048/test_data.json"
    tbee_pred_data = load_data(tbee_model_path)
    tbee_gold_data = load_data("../../data/preprocessed_data/few_fc/test_data.json")
    tbee_gold_data = Preprocessor.choose_spans_by_token_level(tbee_gold_data, "subword")
    score_dict = MetricsCalculator.score(tbee_pred_data, tbee_gold_data)
    fill_scores(score_dict, "BASE")
    
    # tfboys
    # tfboys_model_path = "../../data/res_data/few_fc/re+tfboys+RAIN+TRAIN/2uq6f74s/model_state_dict_22_71.846/test_data.json"
    tfboys_model_path = "../../data/res_data/few_fc/re+tfboys+TFBYB+TRAIN/3s2cljse/model_state_dict_65_76.702/test_data.json"
    tfboys_pred_data = load_data(tfboys_model_path)
    tfboys_gold_data = load_data("../../data/preprocessed_data/few_fc/test_data.json")
    tfboys_gold_data = Preprocessor.choose_spans_by_token_level(tfboys_gold_data, "subword")
    score_dict = MetricsCalculator.score(tfboys_pred_data, tfboys_gold_data)
    fill_scores(score_dict, "tfboys")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    main_res_df = pd.DataFrame(main_res_dict)
    print(main_res_df)






    # tbee_pred_data_path = "../../data/res_data/ace2005_lu/re+tee+RAIN+TRAIN/h87pgw2q/model_state_dict_97_53.333/0_test_data.json"
    # tfee_pred_data_path = "../../data/res_data/ace2005_lu/re+tfboys+RAIN+TRAIN/1bupwzu7/model_state_dict_71_53.844/2_test_data.json"
    # gold_data_path = "../../data/normal_data/ace2005_lu/test_data.json"
    # #
    # tbee_pred_data = load_data(tbee_pred_data_path)
    # # tfee_pred_data = load_data(tfee_pred_data_path)
    # gold_data = load_data(gold_data_path)
    # gold_data = Preprocessor.choose_spans_by_token_level(gold_data, "subword")

    # pred_data = rm_triggers(pred_data)
    # gold_data = rm_triggers(gold_data)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
