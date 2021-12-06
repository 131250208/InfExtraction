from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules.utils import load_data

data_name = "cadec4yelp"
train_path = "../../../data/preprocessed_data/{}/train_data.json".format(data_name)
valid_path = "../../../data/preprocessed_data/{}/valid_data.json".format(data_name)
test_path = "../../../data/preprocessed_data/{}/test_data.json".format(data_name)

train_data = Preprocessor.choose_spans_by_token_level(load_data(train_path), "word")
valid_data = Preprocessor.choose_spans_by_token_level(load_data(valid_path), "word")
test_data = Preprocessor.choose_spans_by_token_level(load_data(test_path), "word")

from InfExtraction.modules.metrics import MetricsCalculator

_, statistics_train = MetricsCalculator.do_additonal_analysis4disc_ent(train_data, train_data)
_, statistics_valid = MetricsCalculator.do_additonal_analysis4disc_ent(valid_data, valid_data)
_, statistics_test = MetricsCalculator.do_additonal_analysis4disc_ent(test_data, test_data)


def filter_stat(statistics):
    statistics_inter = [(k, v) for k, v in statistics.items() if "inter" in k]
    statistics_inter = sorted(statistics_inter, key=lambda x: x[0].split(":")[1])

    statistics_span = [(k, v) for k, v in statistics.items() if "span" in k]
    statistics_span = sorted(statistics_span, key=lambda x: x[0].split(":")[1])
    return statistics_inter, statistics_span


inter_statistics_train, span_statistics_train = filter_stat(statistics_train)
inter_statistics_valid, span_statistics_valid = filter_stat(statistics_valid)
inter_statistics_test, span_statistics_test = filter_stat(statistics_test)
pass