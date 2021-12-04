from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import load_data
from InfExtraction.modules.preprocess import Preprocessor
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots()

# cadec: 353778
dai_pred_file_path = "../../../data/res_data/dai_disc_ner/elmo/cadec4uncbase/353778/test_data.json"
our_pred_file_path = "../../../data/res_data/macd_res/cadec/test_data.json"
gold_file_path = "../../../data/preprocessed_data/cadec4yelp/test_data.json"
save_dir = "../../../data/res_data/analysis/disc_ner/cadec"

# # 13: 50542
# dai_pred_file_path = "../../../data/res_data/dai_disc_ner/elmo/share_13_uncbase/50542/test_data.json"
# our_pred_file_path = "../../../data/res_data/macd_res/share_13/test_data.json"
# gold_file_path = "../../../data/preprocessed_data/share_13_clinic/test_data.json"
# save_dir = "../../../data/res_data/analysis/disc_ner/share_13"

# # 14: 869
# dai_pred_file_path = "../../../data/res_data/dai_disc_ner/elmo/share_14_uncbase/869/test_data.json"
# our_pred_file_path = "../../../data/res_data/macd_res/share_14/test_data.json"
# gold_file_path = "../../../data/preprocessed_data/share_14_clinic/test_data.json"
# save_dir = "../../../data/res_data/analysis/disc_ner/share_14"


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dai_pred_data = load_data(dai_pred_file_path)
our_pred_data = load_data(our_pred_file_path)
gold_data = load_data(gold_file_path)

gold_data = Preprocessor.choose_spans_by_token_level(gold_data, "word")

our_pred_data = sorted(our_pred_data, key=lambda x: x["id"])
dai_pred_data = sorted(dai_pred_data, key=lambda x: x["id"])
gold_data = sorted(gold_data, key=lambda x: x["id"])

for idx, dai_pred_sample in enumerate(dai_pred_data):
    gold_sample4dai = gold_data[idx]
    # gold_sample4our_method = gold_data4our_method[idx]
    our_pred_sample = our_pred_data[idx]
    assert gold_sample4dai["text"] == \
           dai_pred_sample["text"] == our_pred_sample["text"]

# subword span to word span
for idx, sample in enumerate(our_pred_data):
    subwd2word = gold_data[idx]["features"]["subword2word_id"]
    for ent in sample["entity_list"]:
        subwd_sp = ent["tok_span"]
        wd_sp = []
        for i in range(0, len(subwd_sp), 2):
            wd_ids = subwd2word[subwd_sp[i]:subwd_sp[i + 1]]
            try:
                wd_sp.extend([wd_ids[0], wd_ids[-1] + 1])
            except Exception:
                pass  # temp
        ent["tok_span"] = wd_sp

dai_analysis, statistics = MetricsCalculator.do_additonal_analysis4disc_ent(dai_pred_data, gold_data)
our_analysis, _ = MetricsCalculator.do_additonal_analysis4disc_ent(our_pred_data, gold_data)

# span length
span_len = []
f1 = []
method = []
for k, val in statistics.items():
    if "span_len" in k:
        len_ = re.match("span_len: (.*)", k).group(1)
        # span_len.append("{} ({})".format(len_, val))
        span_len.append(len_)
        f1.append(dai_analysis[k][2])
        method.append("Trans$_E$")

        # span_len.append("{} ({})".format(len_, val))
        span_len.append(len_)
        f1.append(our_analysis[k][2])
        method.append("Mac")

f1 = [v * 100 for v in f1]
df_span_len = pd.DataFrame({
    "Method": method,
    "F1 score": f1,
    "Span length": span_len,
})
df_span_len = df_span_len.sort_values(by="Span length")
# sns.set_theme(style="whitegrid")
ax = sns.lineplot(x="Span length", y="F1 score",
             hue="Method",
             data=df_span_len)

# plt.ylim(0, 100)
#plt.title('(a) NYT')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Span length", fontdict={'size': 20})
plt.ylabel("F1 score", fontdict={'size': 20})
# legend
ax.legend_.set_title("")
plt.setp(ax.legend_.get_texts(), fontsize='20')  # for legend text
plt.setp(ax.legend_.get_title(), fontsize='20')
plt.savefig(os.path.join(save_dir, "span_len.pdf"), dpi=200, bbox_inches='tight')
plt.show()

# interval length
inter_len = []
f1 = []
method = []
for k, val in statistics.items():
    if "interval_len" in k:
        len_ = re.match("interval_len: (.*)", k).group(1)
        # inter_len.append("{} ({})".format(len_, val))
        inter_len.append(len_)
        f1.append(dai_analysis[k][2])
        method.append("Trans$_E$")

        # inter_len.append("{} ({})".format(len_, val))
        inter_len.append(len_)
        f1.append(our_analysis[k][2])
        method.append("Mac")

f1 = [v * 100 for v in f1]
df_inter_len = pd.DataFrame({
    "Method": method,
    "F1 score": f1,
    "Interval length": inter_len,
})
df_inter_len = df_inter_len.sort_values(by="Interval length")
# sns.set_theme(style="whitegrid")
ax = sns.lineplot(x="Interval length", y="F1 score",
             hue="Method",
             data=df_inter_len)

# plt.ylim(0, 100)
#plt.title('(a) NYT')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Interval length", fontdict={'size': 20})
plt.ylabel("F1 score", fontdict={'size': 20})
# legend
ax.legend_.set_title("")
plt.setp(ax.legend_.get_texts(), fontsize='20')  # for legend text
plt.setp(ax.legend_.get_title(), fontsize='20')

plt.savefig(os.path.join(save_dir, "inter_len.pdf", ), dpi=200, bbox_inches='tight')
plt.show()

overlap_type = []
method = []
f1 = []
order = []
order_map = {
    "no_overlap": 0,
    "left_overlap": 1,
    "right_overlap": 2,
    "multi_overlap": 3,
}

for k, val in statistics.items():
    if k in {"left_overlap", "right_overlap", "multi_overlap", "no_overlap"}:
        # x = "{} ({})".format(re.sub("_overlap", "", k), val)
        x = re.sub("_overlap", "", k)
        our_f1 = our_analysis[k][2]
        dai_f1 = dai_analysis[k][2]

        overlap_type.append(x)
        method.append("Trans$_E$")
        f1.append(dai_f1)
        order.append(order_map[k])

        overlap_type.append(x)
        method.append("Mac")
        f1.append(our_f1)
        order.append(order_map[k])

f1 = [round(v * 100, 2) for v in f1]
df_overlap = pd.DataFrame({
    "Method": method,
    "F1_score": f1,
    "Overlap type": overlap_type,
    "order": order,
})

df_overlap = df_overlap.sort_values(by="order")

# sns.set_theme(style="whitegrid")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df_overlap, kind="bar",
    x="Overlap type", y="F1_score", hue="Method",
    ci="sd", palette="dark", alpha=1., height=6,
    legend_out=False,
)

g.despine(left=True)
g.set_axis_labels("", "F1 score", fontsize=20)
g.legend.set_title("")
plt.setp(g.legend.get_texts(), fontsize='20')  # for legend text
plt.setp(g.legend.get_title(), fontsize='20')

# plt.ylim(0, 100)
#plt.title('(a) NYT')
# set vals
for i in range(len(df_overlap)):
    row = df_overlap.iloc[i]
    text = row.F1_score if row.F1_score != 0. else ""
    # text = row.F1_score
    # text = "{} ".format(row.F1_score) if row.Method == "Trans$_E$" else " {}".format(row.F1_score)
    ha = "right" if row.Method == "Trans$_E$" else "left"
    g.ax.text(int(row.order), float(row.F1_score), text,
              color="black", horizontalalignment=ha, fontdict={'size': 15})

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(save_dir, "overlap.pdf"), dpi=200, bbox_inches='tight')
plt.show()