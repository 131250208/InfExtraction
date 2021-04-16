from InfExtraction.modules.metrics import MetricsCalculator
from pprint import pprint
import json
from tqdm import tqdm


def load_data(path, total_lines=None):
    filename = path.split("/")[-1]
    try:
        print("loading data: {}".format(filename))
        data = json.load(open(path, "r", encoding="utf-8"))
        if total_lines is not None:
            print("total number is set: {}".format(total_lines))
            data = data[:total_lines]
        print("done! {} samples are loaded!".format(len(data)))
    except json.decoder.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as file_in:
            if total_lines is not None:
                print("total number is set: {}".format(total_lines))
            data = []
            for line in tqdm(file_in, desc="loading data {}".format(filename), total=total_lines):
                data.append(json.loads(line))
                if total_lines is not None and len(data) == total_lines:
                    break
    return data


# # Transition
# pred_path = "../../data/res_data/significance_test/Trans/pred/share_14/pred_data_2.json"
# gold_path = "../../data/res_data/significance_test/Trans/gold/share_14.json"

# # Mac
pred_path = "../../data/res_data/significance_test/Mac/pred/share_13/pred_data_0.json"
gold_path = "../../data/res_data/significance_test/Mac/gold/share_13.json"

pred_data = load_data(pred_path)
gold_data = load_data(gold_path)
cpg_dict = MetricsCalculator.get_ent_cpg_dict(pred_data, gold_data)
prf_dict = {}
for k, cpg in cpg_dict.items():
    prf_dict[k] = MetricsCalculator.get_prf_scores(*cpg)
pprint(prf_dict)