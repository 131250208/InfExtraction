import copy
import re
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from InfExtraction.modules.preprocess import Indexer, Preprocessor


class Tagger(metaclass=ABCMeta):
    @abstractmethod
    def get_tag_size(self):
        pass

    def additional_preprocess(self, data):
        return data

    @abstractmethod
    def get_tag_points(self, sample):
        '''
        This function is for generating tag points

        sample: an example
        return points for tagging
        point: (start_pos, end_pos, tag_id)
        '''
        pass

    @abstractmethod
    def tag(self, data):
        '''
        This function is for generating tag points in batch

        data: examples
        return: data with points
        '''
        pass

    @abstractmethod
    def decode(self, sample, pred_tag):
        '''
        decoding function: to extract results by the predicted tag

        :param sample: an example (to offer text, tok2char_span for decoding)
        :param pred_tag: predicted tag
        :return: predicted example
        '''
        pass

    @abstractmethod
    def decode_batch(self, data, pred_tag_batch):

        '''
        decoding function for batch data, based on decode()
        :param data: examples (to offer text, tok2char_span for decoding)
        :param pred_tag_batch:
        :return:predicted example list
        '''
        pass
    

class HandshakingTaggerRel(Tagger):
    def additional_preprocess(self, data):
        for sample in data:
            fin_ent_list = []

            for rel in sample["relation_list"]:
                # add relation type to entities
                fin_ent_list.append({
                    "text": rel["subject"],
                    "type": rel["predicate"],
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })
                fin_ent_list.append({
                    "text": rel["object"],
                    "type": rel["predicate"],
                    "char_span": rel["obj_char_span"],
                    "tok_span": rel["obj_tok_span"],
                })
                # add default tag to entities
                fin_ent_list.append({
                    "text": rel["subject"],
                    "type": "EXT:DEFAULT",
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })
                fin_ent_list.append({
                    "text": rel["object"],
                    "type": "EXT:DEFAULT",
                    "char_span": rel["obj_char_span"],
                    "tok_span": rel["obj_tok_span"],
                })
            fin_ent_list.extend(sample["entity_list"])
            sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
        return data

    def __init__(self, data):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super().__init__()
        # additional preprocessing
        data = self.additional_preprocess(data)

        # generate entity type and relation type dicts
        rel_type_set = set()
        ent_type_set = set()
        for sample in data:
            # entity type
            ent_type_set |= {ent["type"] for ent in sample["entity_list"]}
            # relation type
            rel_type_set |= {rel["predicate"] for rel in sample["relation_list"]}
        rel_type_set = sorted(rel_type_set)
        ent_type_set = sorted(ent_type_set)
        self.rel2id = {rel: ind for ind, rel in enumerate(rel_type_set)}
        self.ent2id = {ent: ind for ind, ent in enumerate(ent_type_set)}

        self.id2rel = {ind: rel for rel, ind in self.rel2id.items()}

        self.separator = "\u2E80"
        self.rel_link_types = {"SH2OH",  # subject head to object head
                               "OH2SH",  # object head to subject head
                               "ST2OT",  # subject tail to object tail
                               "OT2ST",  # object tail to subject tail
                               "S2O", # won't be used in decoding
                               "O2S" # won't be used in decoding
                              }
        self.tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}

        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent, "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)
        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}

    def get_tag_size(self):
        return len(self.tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            sample["tag_points"] = self.get_tag_points(sample)
        return data

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_points = []
        point_memory_set = set()

        def add_point(point):
            memory = "{},{},{}".format(*point)
            if memory not in point_memory_set:
                matrix_points.append(point)
                point_memory_set.add(memory)

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                add_point(
                    (ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                # add relation points
                for i in range(*subj_tok_span):
                    for j in range(*obj_tok_span):
                        if i <= j:
                            add_point((i, j, self.tag2id[self.separator.join([rel, "S2O"])]))
                        else:
                            add_point((j, i, self.tag2id[self.separator.join([rel, "O2S"])]))

                # add related boundaries
                if subj_tok_span[0] <= obj_tok_span[0]:
                    add_point((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
                else:
                    add_point((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
                if subj_tok_span[1] <= obj_tok_span[1]:
                    add_point((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
                else:
                    add_point((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_points

    def decode(self, sample, predicted_shaking_tag):
        '''
        sample: to provide tok2char_span map and text
        predicted_shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list, ent_list = [], []
        matrix_points = Indexer.shaking_seq2points(predicted_shaking_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        head_ind2entities = {}
        for sp in matrix_points:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1]:
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            ent_list.append(entity)

            head_key = "{},{}".format(ent_type, str(sp[0]))
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_points:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        for sp in matrix_points:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = "{},{}".format(rel, str(sp[0])), "{},{}".format(rel, str(sp[1]))
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = "{},{}".format(rel, str(sp[1])), "{},{}".format(rel, str(sp[0]))
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue

            # all entities start with this subject head
            subj_list = Preprocessor.unique_list(head_ind2entities[subj_head_key])
            # all entities start with this object head
            obj_list = Preprocessor.unique_list(head_ind2entities[obj_head_key])

            # go over all subj-obj pair to check whether the tail link exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    if tail_link_memory not in tail_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                        "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                        "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                        "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                        "predicate": rel,
                    })

        pred_sample = copy.deepcopy(sample)
        # change to predicted relation list and entity list
        pred_sample["relation_list"] = rel_list
        pred_sample["entity_list"] = ent_list

        # res = {
        #     "id": sample_idx,
        #     "text": text,
        #     "tok2char_span": tok2char_span,
        #     "relation_list": rel_list,
        #     "entity_list": ent_list,
        #     # "tok_level_offset": sample["tok_level_offset"],
        #     # "char_level_offset": sample["char_level_offset"],
        # }
        # # these three keys are for span recovering (to original text)
        # if "tok_level_offset" in sample:
        #     res["tok_level_offset"] = sample["tok_level_offset"]
        # if "char_level_offset" in sample:
        #     res["char_level_offset"] = sample["char_level_offset"]
        # if "splits" in sample: # it is a combined sample
        #     res["splits"] = sample["splits"]
        return pred_sample

    def decode_batch(self, sample_list, pred_tag_batch):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tag = pred_tag_batch[ind]
            pred_sample = self.decode(sample, pred_tag)  # decoding
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class HandshakingTaggerEE(HandshakingTaggerRel):
    def additional_preprocess(self, data):
        separator = "_"
        for sample in data:
            # transform event list to relation list and entity list
            fin_ent_list = []
            fin_rel_list = []
            for event in sample["event_list"]:
                fin_ent_list.append({
                    "text": event["trigger"],
                    "type": "EE:{}{}{}".format("Trigger", separator, event["trigger_type"]),
                    "char_span": event["trigger_char_span"],
                    "tok_span": event["trigger_tok_span"],
                })
                for arg in event["argument_list"]:
                    fin_ent_list.append({
                        "text": arg["text"],
                        "type": "EE:{}{}{}".format("Argument", separator, arg["type"]),
                        "char_span": arg["char_span"],
                        "tok_span": arg["tok_span"],
                    })
                    fin_rel_list.append({
                        "subject": arg["text"],
                        "subj_char_span": arg["char_span"],
                        "subj_tok_span": arg["tok_span"],
                        "object": event["trigger"],
                        "obj_char_span": event["trigger_char_span"],
                        "obj_tok_span": event["trigger_tok_span"],
                        "predicate": "EE:{}{}{}".format(arg["type"], separator, event["trigger_type"]),
                    })
            sample["relation_list"] = Preprocessor.unique_list(fin_rel_list)
            # extend original entity list
            if "entity_list" in sample:
                fin_ent_list.extend(sample["entity_list"])
            sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
        data = super(HandshakingTaggerEE, self).additional_preprocess(data)
        return data

    def decode(self, sample, shaking_tag):
        pred_sample = super(HandshakingTaggerEE, self).decode(sample, shaking_tag)
        return {
            **pred_sample,
            "event_list": self._trans2ee(pred_sample["relation_list"], pred_sample["entity_list"])
        }

    def _trans2ee(self, rel_list, ent_list):
        # choose tags with EE:
        new_rel_list, new_ent_list = [], []
        for rel in rel_list:
            if rel["predicate"].split(":")[0] == "EE":
                new_rel = copy.deepcopy(rel)
                new_rel["predicate"] = re.sub(r"EE:", "", new_rel["predicate"])
                new_rel_list.append(new_rel)
        for ent in ent_list:
            if ent["type"].split(":")[0] == "EE":
                new_ent = copy.deepcopy(ent)
                new_ent["type"] = re.sub(r"EE:", "", new_ent["type"])
                new_ent_list.append(new_ent)
        rel_list, ent_list = new_rel_list, new_ent_list

        sepatator = "_"
        trigger_offset2vote = {}
        trigger_offset2trigger_text = {}
        trigger_offset2trigger_char_span = {}
        # get candidate trigger types from relation
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
            trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
            _, event_types = rel["predicate"].split(sepatator)

            if trigger_offset_str not in trigger_offset2vote:
                trigger_offset2vote[trigger_offset_str] = {}
            trigger_offset2vote[trigger_offset_str][event_types] = trigger_offset2vote[trigger_offset_str].get(
                event_types, 0) + 1

        # get candidate trigger types from entity tags
        for ent in ent_list:
            t1, t2 = ent["type"].split(sepatator)
            # assert t1 == "Trigger" or t1 == "Argument"
            if t1 == "Trigger":  # trigger
                event_types = t2
                trigger_span = ent["tok_span"]
                trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_types] = trigger_offset2vote[trigger_offset_str].get(
                    event_types, 0) + 1  # if even, entity type makes the call

        # choose the final trigger type by votes
        tirigger_offset2event_types = {}
        for trigger_offet_str, event_type2score in trigger_offset2vote.items():
            top_score = sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][1]
            winer_event_types = {et for et, sc in event_type2score.items() if sc == top_score}
            # winer_event_types = {sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][0],} # ignore draw
            tirigger_offset2event_types[trigger_offet_str] = winer_event_types  # final event types

        # generate event list
        trigger_offset2event2arguments = {}
        for rel in rel_list:
            trigger_offset = rel["obj_tok_span"]
            argument_role, et = rel["predicate"].split(sepatator)
            trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
            if et not in tirigger_offset2event_types[trigger_offset_str]:  # filter false relations
                continue
            # append arguments
            if trigger_offset_str not in trigger_offset2event2arguments:
                trigger_offset2event2arguments[trigger_offset_str] = {}
            if et not in trigger_offset2event2arguments[trigger_offset_str]:
                trigger_offset2event2arguments[trigger_offset_str][et] = []
            trigger_offset2event2arguments[trigger_offset_str][et].append({
                "text": rel["subject"],
                "type": argument_role,
                "char_span": rel["subj_char_span"],
                "tok_span": rel["subj_tok_span"],
            })
        event_list = []
        for trigger_offset_str, event_types in tirigger_offset2event_types.items():
            for et in event_types:
                arguments = []
                if trigger_offset_str in trigger_offset2event2arguments and \
                        et in trigger_offset2event2arguments[trigger_offset_str]:
                    arguments = trigger_offset2event2arguments[trigger_offset_str][et]

                trigger_offset = trigger_offset_str.split(",")
                event = {
                    "trigger": trigger_offset2trigger_text[trigger_offset_str],
                    "trigger_char_span": trigger_offset2trigger_char_span[trigger_offset_str],
                    "trigger_tok_span": [int(trigger_offset[0]), int(trigger_offset[1])],
                    "trigger_type": et,
                    "argument_list": arguments,
                }
                event_list.append(event)
        return event_list


class MatrixTaggerEE(Tagger):
    def __init__(self, data):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super().__init__()
        # generate unified tag
        tag_set = set()
        self.separator = "\u2E80"
        for sample in data:
            tag_triplets = self._get_tags(sample)
            tag_set |= {t[-1] for t in tag_triplets}

        self.tag2id = {tag: ind for ind, tag in enumerate(sorted(tag_set))}
        self.id2tag = {id_: t for t, id_ in self.tag2id.items()}

    def get_tag_size(self):
        return len(self.tag2id)

    def _get_tags(self, sample):
        tag_list = []
        for event in sample["event_list"]:
            event_type = event["trigger_type"]
            pseudo_argument = {
                "type": "Trigger",
                "tok_span": event["trigger_tok_span"],
            }
            argument_list = [pseudo_argument, ] + event["argument_list"]
            for guide_arg in argument_list:
                for arg in argument_list:
                    arg_type = arg["type"]
                    ea_tag = "{}{}{}".format(event_type, self.separator, arg_type)
                    for i in range(*guide_arg["tok_span"]):
                        for j in range(*arg["tok_span"]):
                            pos_tag = "I"
                            if j == arg["tok_span"][0]:
                                pos_tag = "B"
                            eap_tag = "{}{}{}".format(ea_tag, self.separator, pos_tag)
                            tag_list.append([i, j, eap_tag])
        return tag_list

    def get_tag_points(self, sample):
        tag_list = self._get_tags(sample)
        for tag in tag_list:
            tag[-1] = self.tag2id[tag[-1]]
        return tag_list

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            sample["tag_points"] = self.get_tag_points(sample)
        return data

    def decode(self, sample, predicted_matrix_tag):
        matrix_points = Indexer.matrix2points(predicted_matrix_tag)
        tags = [[p[0], p[1], self.id2tag[p[2]]] for p in matrix_points]
        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # decoding
        # ...

        # event = {
        #     "trigger": None, # "trigger_text
        #     "trigger_char_span": None,
        #     "trigger_tok_span": None,
        #     "trigger_type": None,
        #     "argument_list": [
        #         {
        #             "text": None, # argument text,
        #             "tok_span": None,
        #             "char_span": None,
        #         }
        #     ],
        # }
        event_list = [] # predicted event list

        pred_sample = copy.deepcopy(sample)
        # change to the predicted one, if not, will use ground truth to score
        pred_sample["event_list"] = event_list
        return pred_sample

    def decode_batch(self, sample_list, pred_tag_batch):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tag = pred_tag_batch[ind]
            pred_sample = self.decode(sample, pred_tag)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list
