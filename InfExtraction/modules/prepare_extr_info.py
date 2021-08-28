# if add pos tags, ner tags, dependency relations
# parse texts
import os
from InfExtraction.modules.utils import load_data, save_as_json_lines, MyLargeFileReader, MyLargeJsonlinesFileReader
from tqdm import tqdm
from ddparser import DDParser


def gen_ddp_data(in_file_path):
    in_file_id = os.stat(in_file_path).st_ino
    if not os.path.exists("../../data/cache"):
        os.mkdir("../../data/cache")

    cache_path = "../../data/cache/parse_cache_{}.jsonlines".format(in_file_id)

    if os.path.exists(cache_path):
        parse_results = load_data(cache_path)
        print(">>>>>>>>>>>>>> loaded parse res >>>>>>>>>>>>>>>>>>\n{}".format(in_file_path))
    else:
        parse_results = []
        ddp = DDParser(use_pos=True, buckets=True)
        data_jsreader = MyLargeJsonlinesFileReader(MyLargeFileReader(in_file_path))
        for sample in tqdm(data_jsreader.get_jsonlines_generator(), desc="ddp parse"):
            parse_results.append(ddp.parse(sample["text"]))

        save_as_json_lines(parse_results, cache_path)
    return parse_results

# res = gen_ddp_data("../../data/ori_data/duee_fin_comp2021_bk/test_data_1.json")