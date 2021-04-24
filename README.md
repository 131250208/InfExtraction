# Information Extraction
## Requirements

```
torch
flair
allennlp
transformers
wandb
stanza
networkx
pattern
```
**Note: `pattern.en.lemma` is required for the evaluation of OIE (Open Information Extraction) task. 
To avoid some problems and save your time, please follow these steps to install `pattern`**:

Download `pattern` and edit `setup.py`
```
git clone https://github.com/clips/pattern.git
cd pattern
vim setup.py
```
Comment `mysqlclient` since it is optional.
```python
install_requires = [
    "future",
    "backports.csv",
#    "mysqlclient",
    "beautifulsoup4",
    "lxml",
    "feedparser",
    "pdfminer" if sys.version < "3" else "pdfminer.six",
    "numpy",
    "scipy",
    "nltk",
    "python-docx",
    "cherrypy",
    "requests"
],
```
Install `pattern`
```
python setup.py install
```
To use `pattern.en.lemma`, you need to download several copora from `nltk`:
```
>>> import nltk
>>> nltk.download()
```
Go to the Corpora tab and download the `wordnet`, `wordnet_ic`, `sentiwordnet`.
You can also download them by our [link](https://drive.google.com/file/d/1wYIHRCyuuPkwaPB2e8Zy8MSMT2pCjBVI/view?usp=sharing) and put them under:
```
Windows C:\Users\<your name>\AppData\Roaming\nltk_data\corpora
Linux /home/<your name>/nltk_data/corpora
```
Because `pattern` is based on python 3.6, if you use python >= 3.7, you need to run this patch function after import it. 
```python
from pattern import text
import functools

def patch_pattern():
    original_read = text._read

    @functools.wraps(original_read)
    def patched_read(*args, **kwargs):
        try:
            for r in original_read(*args, **kwargs):
                yield r
        except RuntimeError:
            pass
    text._read = patched_read
```

## Usage
### Data format
We provided some codes for converting several data formats to ours: 
```
casrel, etl_span, raw_nyt, duie_1, duie_2, duee_1, duee_fin
```
Check out [TPlinker](https://github.com/131250208/TPlinker-joint-extraction) for the description of `casrel`, `etl_span`, and `raw_nyt`. 
And `du*` are the Qian Yan datasets provided by Baidu. We provide a fast [link](https://drive.google.com/drive/folders/13cXK0KZYmyhpVKa75m7vcnnlRzZxwSGk?usp=sharing) to download these datasets.

For other datasets, you need to transform your data to our format (check out `data_example/data.json` for more details):
```json
{
    "id": "<id>",
    "text": "<text>",
    "word_list": "<*optional: word list>",
    "word2char_span": "<*optional: a list mapping word level offset to character level offset>",
    "ner_tag_list": "<*optional: tag list for named entities>",
    "pos_tag_list": "<*optional: position tag list>",
    "dependency_list": "<*optional: dependency relation between words>",
    "entity_list":[
        {
            "text": "<entity text>",
            "type": "<entity type>",
            "char_span": "<the character level offset of this entity>, e.g.: [13, 15], or [13, 15, 20, 22](only for discontinuous NER task)"
        }
    ],
    "relation_list":[
        {
            "subject": "<subject text>",
            "subj_char_span": "<the character level offset of the subject>",
            "predicate": "<relation type, e.g. 'contains'>",
            "object": "<object text>",
            "obj_char_span": "<the character level offset of the object>"
        }
    ],
    "event_list":[
        {
            "event_type": "<event type>",
            "trigger": "<trigger text>",
            "trigger_char_span": "<the character level offset of the trigger>",
            "argument_list":[
                {
                    "text": "<argument text>",
                    "type": "<argument role>",
                    "char_span": "<the character level offset of this argument>, [13, 15] or [[13, 15], [26, 29]]"
                }
            ]
        }
    ],
    "open_spo_list":[
        [
            {
                    "text": "<argument text>",
                    "type": "<argument role>",
                    "char_span": "<the character level offset of this argument>"
            }
        ]
    ]
}
```
Note:
1. The `ner_tag_list`, the `pos_tag_list`, and the `dependency_list` are optional, you can set up if your model needs these features. 
2. The `word_list` and the `word2char_span` are also optional. If not provided, they will be auto-generated in the preprocessing stage. 
3. As for `entity_list`, `relation_list`, `event_list`, and `open_spo_list`, check out the following table.
4. In a discontinuous NER task, the length of `*_span` in `entity_list` can be larger than 2, for example, [13, 15, 20, 22].
5. In an event extraction task, `*_span` can be a list of span, e.g., [[13, 15], [26, 29]]. This is for the datasets that do not provide
argument offsets. The list of spans are auto searched from the text. In this way, some offset-must metrics would not be considered.

|Task|entity_list|relation_list|event_list|open_spo_list|
|:---|---|---|---|---|
|NER|must|no|no|no|
|RE|optional|must|no|no|
|EE|optional|optional|must|no|
|OIE|optional|optional|no|must|

### Preprocess
Put the datasets under `data/ori_data` and do preprocessing by `preprocess.py` under `InfExtraction/workflows`:

Edit `settings_preprocess.py`:
```python
data_in_dir = "../../data/ori_data/duie_comp2021"
data_out_dir = "../../data/normal_data/duie_comp2021"

language = "ch"  # en, ch

# it is valid only if "word_list" or "word2char_span" are not provided
word_tokenizer_type = "normal_chinese"  # white (for English), normal_chinese (for Chinese);

pretrained_model_tokenizer_path = "../../data/pretrained_models/chinese_roberta_wwm_ext_pytorch"
do_lower_case = True  # transform to lower case and rm accents, only for bert tokenizer and lower the subword_list,
                    # it does not change the original text or word_list,
                    # set True if use uncased BERT

# We provided some codes for converting several data format to ours: 
# casrel (webnlg_star, nyt_star), etl_span (webnlg), raw_nyt (nyt), duie_1, duie_2, duee_1, duee_fin
# For other datasets, you need to transform your data to our format as aforementioned and set "ori_data_format" to "normal".
ori_data_format = "duie_2"

add_id = True  # Set True if "id" is not provided in the data

add_char_span = True  # Set True for data without character level spans
ignore_subword_match = False # It is valid only if add_char_span is set to True. Add whitespaces around entities when matching and adding character level spans,
# e.g. if ignore_subword_match is set true, " home " will not match the subword "home" in "hometown"
# Note that it should be set to False when preprocessing Chinese datasets

# Only for word embedding, it does not matter if only use bert
max_word_dict_size = 50000  # the max size of word2id dict
min_word_freq = 1
```

Run
```
python preprocess.py
```

### Train and Evaluation

Make sure your datasets are put under `data/normal_data`

If use BERT, put it under `data/pretrained_models` and set `pretrained_model_name`.

Set the BERT configuration file `config.json`:
```
"output_hidden_states": true,
"output_attentions": true,
```

If use word embedding, put it under `data/pretrained_emb` and set `pretrained_emb_name`. 
We support both `*.txt` and `*.bin`.

Set `settings_re.py` for training:
```python
stage = "train" 
exp_name = "duie_comp2021" # the folder name of your datasets
pretrained_model_name = "chinese_roberta_wwm_ext_pytorch" # bert name
test_path_list = []  # glob("{}/*test*.json".format(os.path.join(data_in_dir, exp_name)))
...
```
If `test_path_list` is not empty, will also do validation on test sets.

Set `settings_re.py` for evaluation:
```python
stage = "inference" 
exp_name = "duie_comp2021"
pretrained_model_name = "chinese_roberta_wwm_ext_pytorch"
test_path_list = glob("{}/*test*.json".format(os.path.join(data_in_dir, exp_name)))
...

# valid only if stage == "inference" 
model_dir_for_test = "./wandb"  # "./default_log_dir", "./wandb", model state dicts are all saved under this path
target_run_ids = ["eyu8cm6x", ]  # run id
top_k_models = 1  # use top k models (at validation set) to eval
metric4testing = "rel_exact_text_f1"  # by which metric to choose top k models 
cal_scores = True  # set False if all the test sets are not annotated, then only output the result files without scoring.
...
```
Do evaluation on the datasets in `test_path_list`. The result data will be saved under `data/res_data` after evaluation.

Run `train_valid.py` for either training or evaluation:
```
python train_valid.py -s settings_re
```

## Tasks
### Named Entity Recognition
```
python train_valid.py -s settings_span_ner
```
### Joint Extraction of Entity and Relation
```
python train_valid.py -s settings_re
```
### Event Extraction
Trigger-based EE
```
python train_valid.py -s settings_re+tee
```
Trigger-free EE
```
python train_valid.py -s settings_re+tfboys
```
### Open Information Extraction
```
python train_valid.py -s settings_re+oie
```

### Discontinuous NER
```
python train_valid.py -s settings_re+disc
```

Download datasets
* CADEC: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
* ShARe 13: https://physionet.org/content/shareclefehealth2013/1.0/
* ShARe 14: https://physionet.org/content/shareclefehealth2014task2/1.0/
Preprocess the datasets by the open-sourced code of [Trans](https://github.com/daixiangau/acl2020-transition-discontinuous-ner).

Transform data
```python
import os
from  InfExtraction.work_flows.format_conv import trans_daixiang_data
data_in_dir = "<data_in_dir>"
train_filename = "train.txt"
valid_filename = "dev.txt"
test_filename = "test.txt"

train_path = os.path.join(data_in_dir, train_filename)
valid_path = os.path.join(data_in_dir, valid_filename)
test_path = os.path.join(data_in_dir, test_filename)

train_data = trans_daixiang_data(train_path)
valid_data = trans_daixiang_data(valid_path)
test_data = trans_daixiang_data(test_path)
```

Download in-field BERTs
Find links in the papers:

Clinical BERT: https://www.aclweb.org/anthology/W19-1909/

YelpBERT: https://www.aclweb.org/anthology/2020.findings-emnlp.151/


### statistics
Run `InfExtraction/workflows/others/statistics.py` 

### further analysis
Run `InfExtraction/workflows/further_analysis.py` 
