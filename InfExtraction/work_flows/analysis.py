from transformers import BertTokenizerFast
from InfExtraction.modules.utils import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#
tokenizer = BertTokenizerFast.from_pretrained("../../data/pretrained_models/macbert-base", do_lower_case=False)
train_data = load_data("../../data/ori_data/few_fc/train_data.json")
valid_data = load_data("../../data/ori_data/few_fc/valid_data.json")
test_data = load_data("../../data/ori_data/few_fc/test_data.json")

token_num_list = []
for sample in train_data + valid_data + test_data:
    tokens = tokenizer.tokenize(sample["text"], add_special_tokens=False)
    token_num_list.append(len(tokens))

df = pd.DataFrame({
    "token_num": token_num_list,
})

sns.histplot(data=df, x="token_num")
plt.show()
