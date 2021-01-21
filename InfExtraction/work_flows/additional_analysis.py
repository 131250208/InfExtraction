from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import load_data
from InfExtraction.modules.preprocess import Preprocessor

pred_file_path = "../../data/res_data/share_14/test_data.json"
gold_file_path = "../../data/normal_data/share_14/test_data.json"

pred_data = load_data(pred_file_path)
gold_data = load_data(gold_file_path)

token_level = "subword"
# pred_data = Preprocessor.choose_spans_by_token_level(pred_data, token_level)
gold_data = Preprocessor.choose_spans_by_token_level(gold_data, token_level)

pred_data = sorted(pred_data, key=lambda x: x["id"])
gold_data = sorted(gold_data, key=lambda x: x["id"])

res, statistics = MetricsCalculator.do_additonal_analysis4disc_ent(pred_data, gold_data)