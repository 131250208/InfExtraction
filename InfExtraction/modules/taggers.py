import copy
import re
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from InfExtraction.modules.preprocess import Indexer, Preprocessor


class Tagger(metaclass=ABCMeta):
    @classmethod
    def additional_preprocess(cls, data):
        return data

    @abstractmethod
    def get_tag_size(self):
        pass

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
    def decode(self, sample, pred_tags):
        '''
        decoding function: to extract results by the predicted tag

        :param sample: an example (to offer text, tok2char_span for decoding)
        :param pred_tags: predicted tag id tensors converted from the outputs of the forward function,
                          it is a tuple or a single tag tensor
        :return: predicted example
        '''
        pass

    # @abstractmethod
    # def decode_batch(self, data, batch_pred_tags):
    #     '''
    #     decoding function for batch data, based on decode()
    #     :param data: examples (to offer text, tok2char_span for decoding)
    #     :param batch_pred_tags:
    #     :return:predicted example list
    #     '''
    #     pass

    def decode_batch(self, sample_list, batch_pred_tags):
        pred_sample_list = []
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            pred_tags = [batch_pred_tag[ind] for batch_pred_tag in batch_pred_tags]
            pred_sample = self.decode(sample, pred_tags)  # decoding one sample
            pred_sample_list.append(pred_sample)
        return pred_sample_list


class HandshakingTagger4TPLPlus(Tagger):
    @classmethod
    def additional_preprocess(cls, data):
        for sample in data:
            fin_ent_list = sample["entity_list"] if "entity_list" in sample else []
            fin_rel_list = sample["relation_list"]

            for rel in fin_rel_list:
                # add additional types to entities
                fin_ent_list.append({
                    "text": rel["subject"],
                    "type": "REL:{}".format(rel["predicate"]),
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })
                fin_ent_list.append({
                    "text": rel["object"],
                    "type": "REL:{}".format(rel["predicate"]),
                    "char_span": rel["obj_char_span"],
                    "tok_span": rel["obj_tok_span"],
                })
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
            sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)

            # add additional relations btw nested entities
            for ent_i in sample["entity_list"]:
                for ent_j in sample["entity_list"]:
                    if (ent_i["tok_span"][1] - ent_i["tok_span"][0]) < (ent_j["tok_span"][1] - ent_j["tok_span"][0]) \
                            and ent_i["tok_span"][0] >= ent_j["tok_span"][0] \
                            and ent_i["tok_span"][1] <= ent_j["tok_span"][1]:
                        fin_rel_list.append({
                            "subject": ent_i["text"],
                            "subj_char_span": ent_i["char_span"],
                            "subj_tok_span": ent_i["tok_span"],
                            "object": ent_j["text"],
                            "obj_char_span": ent_j["char_span"],
                            "obj_tok_span": ent_j["tok_span"],
                            "predicate": "EXT:NESTED_IN",
                        })
                        ent_i_cp = copy.deepcopy(ent_i)
                        ent_i_cp["type"] = "REL:EXT:NESTED_IN"
                        fin_ent_list.append(ent_i_cp)

                        ent_j_cp = copy.deepcopy(ent_j)
                        ent_j_cp["type"] = "REL:EXT:NESTED_IN"
                        fin_ent_list.append(ent_j_cp)

            sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
            sample["relation_list"] = Preprocessor.unique_list(fin_rel_list)
        return data

    def __init__(self, data):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''
        super().__init__()
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
                               "S2O",  # won't be used in decoding
                               "O2S"  # won't be used in decoding
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
                    (ent["tok_span"][0], ent["tok_span"][1] - 1,
                     self.tag2id[self.separator.join([ent["type"], "EH2ET"])]))

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
                    add_point(
                        (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
                else:
                    add_point(
                        (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_points

    def decode(self, sample, pred_tags):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''
        rel_list, ent_list = [], []
        predicted_shaking_tag = pred_tags[0]
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
                subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(sp[0])), "REL:{},{}".format(rel, str(sp[1]))
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(sp[1])), "REL:{},{}".format(rel, str(sp[0]))
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
        # filter extra relations
        pred_sample["relation_list"] = [rel for rel in rel_list if "EXT:" not in rel["predicate"]]
        # filter extra entities
        ent_type_set = {ent["type"] for ent in ent_list}
        ent_types2filter = {"REL:"}
        if len(ent_type_set) == 1 and list(ent_type_set)[0] == "EXT:DEFAULT":
            pass
        else:
            ent_types2filter.add("EXT:")
        filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = [ent for ent in ent_list if re.search(filter_pattern, ent["type"]) is None]
        return pred_sample


class HandshakingTagger4TPLPP(HandshakingTagger4TPLPlus):
    # @classmethod
    # def additional_preprocess(cls, data):
    #     for sample in data:
    #         fin_ent_list = []
    #
    #         for rel in sample["relation_list"]:
    #             # add relation type to entities
    #             fin_ent_list.append({
    #                 "text": rel["subject"],
    #                 "type": rel["predicate"],
    #                 "char_span": rel["subj_char_span"],
    #                 "tok_span": rel["subj_tok_span"],
    #             })
    #             fin_ent_list.append({
    #                 "text": rel["object"],
    #                 "type": rel["predicate"],
    #                 "char_span": rel["obj_char_span"],
    #                 "tok_span": rel["obj_tok_span"],
    #             })
    #
    #             # add default tag to entities
    #             fin_ent_list.append({
    #                 "text": rel["subject"],
    #                 "type": "EXT:DEFAULT",
    #                 "char_span": rel["subj_char_span"],
    #                 "tok_span": rel["subj_tok_span"],
    #             })
    #             fin_ent_list.append({
    #                 "text": rel["object"],
    #                 "type": "EXT:DEFAULT",
    #                 "char_span": rel["obj_char_span"],
    #                 "tok_span": rel["obj_tok_span"],
    #             })
    #
    #         if "entity_list" in sample:
    #             fin_ent_list.extend(sample["entity_list"])
    #         sample["entity_list"] = Preprocessor.unique_list(fin_ent_list)
    #     return data

    def __init__(self, data):
        '''
        :param data: all data, used to generate entity type and relation type dicts
        '''

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

        self.separator = "\u2E80"
        self.rel_link_types = {"SH2OH",  # subject head to object head
                               "OH2SH",  # object head to subject head
                               "ST2OT",  # subject tail to object tail
                               "OT2ST",  # object tail to subject tail
                               "S2O",  # won't be used in decoding
                               "O2S"  # won't be used in decoding
                               }
        self.rel_tags = {self.separator.join([rel, lt]) for rel in self.rel2id.keys() for lt in self.rel_link_types}
        self.ent_tags = {self.separator.join([ent, "EH2ET"]) for ent in self.ent2id.keys()}

        self.rel_tag2id = {t: idx for idx, t in enumerate(sorted(self.rel_tags))}
        self.id2rel_tag = {idx: t for t, idx in self.rel_tag2id.items()}

        self.ent_tag2id = {t: idx for idx, t in enumerate(sorted(self.ent_tags))}
        self.id2ent_tag = {idx: t for t, idx in self.ent_tag2id.items()}

    def get_tag_size(self):
        return len(self.ent_tag2id), len(self.rel_tag2id)

    def tag(self, data):
        for sample in tqdm(data, desc="tagging"):
            ent_points, rel_points = self.get_tag_points(sample)
            sample["ent_points"] = ent_points
            sample["rel_points"] = rel_points
        return data

    def get_tag_points(self, sample):
        '''
        matrix_points: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        ent_matrix_points, rel_matrix_points = [], []

        if "entity_list" in sample:
            for ent in sample["entity_list"]:
                point = (ent["tok_span"][0], ent["tok_span"][1] - 1,
                         self.ent_tag2id[self.separator.join([ent["type"], "EH2ET"])])
                ent_matrix_points.append(point)

        if "relation_list" in sample:
            for rel in sample["relation_list"]:
                subj_tok_span = rel["subj_tok_span"]
                obj_tok_span = rel["obj_tok_span"]
                rel = rel["predicate"]

                # add relation points
                for i in range(*subj_tok_span):
                    for j in range(*obj_tok_span):
                        point = (i, j, self.rel_tag2id[self.separator.join([rel, "S2O"])])
                        rel_matrix_points.append(point)
                        point = (j, i, self.rel_tag2id[self.separator.join([rel, "O2S"])])
                        rel_matrix_points.append(point)

                # add related boundaries
                rel_matrix_points.append(
                    (subj_tok_span[0], obj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "SH2OH"])]))
                rel_matrix_points.append(
                    (obj_tok_span[0], subj_tok_span[0], self.rel_tag2id[self.separator.join([rel, "OH2SH"])]))
                rel_matrix_points.append(
                    (subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "ST2OT"])]))
                rel_matrix_points.append(
                    (obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.rel_tag2id[self.separator.join([rel, "OT2ST"])]))

        return Preprocessor.unique_list(ent_matrix_points), Preprocessor.unique_list(rel_matrix_points)

    def decode(self, sample, pred_tags):
        '''
        sample: to provide tok2char_span map and text
        pred_tags: predicted tags
        '''
        rel_list, ent_list = [], []
        pred_ent_tag, pred_rel_tag = pred_tags[0], pred_tags[1]
        ent_points = Indexer.shaking_seq2points(pred_ent_tag)
        rel_points = Indexer.matrix2points(pred_rel_tag)

        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # entity
        head_ind2entities = {}
        for pt in ent_points:
            ent_tag = self.id2ent_tag[pt[2]]
            ent_type, link_type = ent_tag.split(self.separator)
            # for an entity, the start position can not be larger than the end pos.
            assert link_type == "EH2ET" and pt[0] <= pt[1]

            char_span_list = tok2char_span[pt[0]:pt[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [pt[0], pt[1] + 1],
                "char_span": char_sp,
            }
            ent_list.append(entity)

            head_key = "{},{}".format(ent_type, str(pt[0]))
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)

        # tail link
        tail_link_memory_set = set()
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(pt[0]), str(pt[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(pt[1]), str(pt[0])])
                tail_link_memory_set.add(tail_link_memory)
            else:
                continue

        # head link
        for pt in rel_points:
            tag = self.id2rel_tag[pt[2]]
            rel, link_type = tag.split(self.separator)

            if link_type == "SH2OH":
                subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(pt[0])), "REL:{},{}".format(rel, str(pt[1]))
            elif link_type == "OH2SH":
                subj_head_key, obj_head_key = "REL:{},{}".format(rel, str(pt[1])), "REL:{},{}".format(rel, str(pt[0]))
            else:
                continue

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key or obj_head_key
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

        # filter extra relations
        pred_sample["relation_list"] = [rel for rel in rel_list if "EXT:" not in rel["predicate"]]
        # filter extra entities
        ent_type_set = {ent["type"] for ent in ent_list}
        ent_types2filter = {"REL:"}
        if len(ent_type_set) == 1 and list(ent_type_set)[0] == "EXT:DEFAULT":
            pass
        else:
            ent_types2filter.add("EXT:")
        filter_pattern = "({})".format("|".join(ent_types2filter))
        pred_sample["entity_list"] = [ent for ent in ent_list if re.search(filter_pattern, ent["type"]) is None]

        return pred_sample


def create_rebased_ee_tagger(base_class):
    class REBasedEETagger(base_class):
        def __init__(self, data):
            super(REBasedEETagger, self).__init__(data)
            self.event_type2arg_rols = {}
            for sample in data:
                for event in sample["event_list"]:
                    event_type = event["trigger_type"]
                    for arg in event["argument_list"]:
                        if event_type not in self.event_type2arg_rols:
                            self.event_type2arg_rols[event_type] = set()
                        self.event_type2arg_rols[event_type].add(arg["type"])

        @classmethod
        def additional_preprocess(cls, data):
            separator = "\u2E82"
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
            data = super().additional_preprocess(data)
            return data

        def decode(self, sample, pred_outs):
            pred_sample = super(REBasedEETagger, self).decode(sample, pred_outs)
            pred_sample["event_list"] = self._trans2ee(pred_sample["relation_list"], pred_sample["entity_list"])
            # filter extra entities and relations
            pred_sample["entity_list"] = [ent for ent in pred_sample["entity_list"] if "EE:" not in ent["type"]]
            pred_sample["relation_list"] = [rel for rel in pred_sample["relation_list"] if "EE:" not in rel["predicate"]]
            return pred_sample

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

            # decoding
            sepatator = "\u2E82"
            trigger_offset2vote = {}
            trigger_offset2trigger_text = {}
            trigger_offset2trigger_char_span = {}

            # get candidate trigger types from relations
            for rel in rel_list:
                trigger_offset = rel["obj_tok_span"]
                trigger_offset_str = "{},{}".format(trigger_offset[0], trigger_offset[1])
                trigger_offset2trigger_text[trigger_offset_str] = rel["object"]
                trigger_offset2trigger_char_span[trigger_offset_str] = rel["obj_char_span"]
                _, event_type = rel["predicate"].split(sepatator)

                if trigger_offset_str not in trigger_offset2vote:
                    trigger_offset2vote[trigger_offset_str] = {}
                trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                    event_type, 0) + 1

            # get candidate trigger types from entity tags
            for ent in ent_list:
                t1, t2 = ent["type"].split(sepatator)
                if t1 == "Trigger":
                    event_type = t2
                    trigger_span = ent["tok_span"]
                    trigger_offset_str = "{},{}".format(trigger_span[0], trigger_span[1])
                    trigger_offset2trigger_text[trigger_offset_str] = ent["text"]
                    trigger_offset2trigger_char_span[trigger_offset_str] = ent["char_span"]
                    if trigger_offset_str not in trigger_offset2vote:
                        trigger_offset2vote[trigger_offset_str] = {}
                    trigger_offset2vote[trigger_offset_str][event_type] = trigger_offset2vote[trigger_offset_str].get(
                        event_type, 0) + 1

            # choose the final trigger type by votes
            trigger_offset2event_types = {}
            for trigger_offet_str, event_type2score in trigger_offset2vote.items():
                # # choose types with the top score
                # top_score = sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][1]
                # winer_event_types = {et for et, sc in event_type2score.items() if sc == top_score}

                # ignore draw, choose only the first type
                # winer_event_types = {sorted(event_type2score.items(), key=lambda x: x[1], reverse=True)[0][0],}

                # save all event types
                winer_event_types = set(event_type2score.keys())

                trigger_offset2event_types[trigger_offet_str] = winer_event_types  # final event types

            # aggregate arguments by event type and trigger_offset
            trigger_offset2event2arguments = {}
            for rel in rel_list:
                trigger_offset = rel["obj_tok_span"]
                argument_role, et = rel["predicate"].split(sepatator)
                trigger_offset_str = "{},{}".format(*trigger_offset)
                if et not in trigger_offset2event_types[trigger_offset_str]:  # filter false relations
                    continue
                # append arguments
                if trigger_offset_str not in trigger_offset2event2arguments:
                    trigger_offset2event2arguments[trigger_offset_str] = {}
                if et not in trigger_offset2event2arguments[trigger_offset_str]:
                    trigger_offset2event2arguments[trigger_offset_str][et] = []

                trigger_offset2event2arguments[trigger_offset_str][et].append({
                    "text": rel["subject"],
                    "type": argument_role,
                    "event_type": et,
                    "char_span": rel["subj_char_span"],
                    "tok_span": rel["subj_tok_span"],
                })

            if len(trigger_offset2event_types) == 1:
                for trig_offset_str, event_types in trigger_offset2event_types.items():
                    if len(event_types) == 1:
                        et = list(event_types)[0]
                        for ent in ent_list:
                            t1, t2 = ent["type"].split(sepatator)
                            if t1 == "Argument":
                                arg_role = t2
                                if arg_role not in self.event_type2arg_rols[et]:
                                    continue

                                if trig_offset_str not in trigger_offset2event2arguments:
                                    trigger_offset2event2arguments[trig_offset_str] = {}
                                if et not in trigger_offset2event2arguments[trig_offset_str]:
                                    trigger_offset2event2arguments[trig_offset_str][et] = []

                                trigger_offset2event2arguments[trig_offset_str][et].append({
                                    "text": ent["text"],
                                    "type": arg_role,
                                    "event_type": et,
                                    "char_span": ent["char_span"],
                                    "tok_span": ent["tok_span"],
                                })

            # generate event list
            event_list = []
            for trigger_offset_str, event_types in trigger_offset2event_types.items():
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
                        "argument_list": Preprocessor.unique_list(arguments),
                    }
                    event_list.append(event)
            return event_list

    return REBasedEETagger


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
        event_list = sample["event_list"]

        for event in event_list:
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

    def decode(self, sample, pred_tags):
        predicted_matrix_tag = pred_tags[0]
        matrix_points = Indexer.matrix2points(predicted_matrix_tag)

        # matrix_points = [(58, 58, 87), (58, 59, 88), (58, 60, 88), (58, 65, 89), (58, 68, 91), (58, 75, 82), (58, 82, 84),
        #                  (59, 58, 87), (59, 59, 88), (59, 60, 88), (59, 65, 89), (59, 68, 91), (59, 75, 82), (59, 82, 84),
        #                  (60, 58, 87), (60, 59, 88), (60, 60, 88), (60, 65, 89), (60, 68, 91), (60, 75, 82), (60, 82, 84),
        #                  (65, 58, 87), (65, 59, 88), (65, 60, 88), (65, 65, 89), (65, 68, 91), (65, 75, 82), (65, 82, 84),
        #                  (68, 58, 87), (68, 59, 88), (68, 60, 88), (68, 65, 89), (68, 68, 91), (68, 75, 82), (68, 82, 84),
        #                  (74, 74, 89), (74, 75, 91),
        #                  (75, 74, 89), (75, 75, 91),
        #                  (75, 58, 87), (75, 59, 88), (75, 60, 88), (75, 65, 89), (75, 68, 91), (75, 75, 82), (75, 82, 84),
        #                  (82, 58, 87), (82, 59, 88), (82, 60, 88), (82, 65, 89), (82, 68, 91), (82, 75, 82), (82, 82, 84)]
        # matrix_points = set([(7, 7, 26), (7, 9, 36), (7, 24, 30),
        #                  (9, 7, 26), (9, 9, 36), (9, 24, 30),
        #                  (24, 7, 26), (24, 9, 36), (24, 24, 30),
        #
        #                  (7, 7, 32), (7, 19, 36), (7, 24, 30),
        #                  (19, 7, 32), (19, 19, 36), (19, 24, 30),
        #                  (24, 7, 32), (24, 19, 36), (24, 24, 30),
        #
        #                  (7, 7, 82), (7, 11, 89), (7, 24, 85),
        #                  (11, 7, 82), (11, 11, 89), (11, 24, 85),
        #                  (24, 7, 82), (24, 11, 89), (24, 24, 85),
        #
        #                  (70, 70, 89), (70, 81, 84), (70, 84, 91),
        #                  (81, 70, 89), (81, 81, 84), (81, 84, 91),
        #                  (84, 70, 89), (84, 81, 84), (84, 84, 91),
        #
        #                  (74, 74, 157), (74, 81, 152), (74, 84, 159),
        #                  (81, 74, 157), (81, 81, 152), (81, 84, 159),
        #                  (84, 74, 157), (84, 81, 152), (84, 84, 159),
        #
        #                  (79, 79, 36), (79, 81, 28), (79, 84, 32), (79, 87, 32),
        #                  (81, 79, 36), (81, 81, 28), (81, 84, 32), (81, 87, 32),
        #                  (84, 79, 36), (84, 81, 28), (84, 84, 32), (84, 87, 32),
        #                  (87, 79, 36), (87, 81, 28), (87, 84, 32), (87, 87, 32)])

        # tags = [[p[0], p[1], self.id2tag[p[2]]] for p in matrix_points]
        tags = matrix_points
        sample_idx, text = sample["id"], sample["text"]
        tok2char_span = sample["features"]["tok2char_span"]

        # decoding

        center_tags = []
        for tag in tags:
            if tag[0] == tag[1]:
                center_tags.append(tag)

        event_pieces = []
        num_centers = len(center_tags)
        index = 0

        def center_match(ct_1, ct_2):
            # return predicted_matrix_tag[ct_1[0],ct_2[0],ct_2[2]].item() == 1
            if ct_1 == ct_2:
                return True
            if abs(ct_1[2] - ct_2[2]) > 20:
                return False
            left_match, right_match = False, False
            for tag in tags:
                if tag[0] == ct_1[0] and tag[1] == ct_2[0] and tag[2] == ct_2[2] and tag[0] != tag[1]:
                    # print(tag)
                    left_match = True
                if tag[0] == ct_2[0] and tag[1] == ct_1[0] and tag[2] == ct_1[2] and tag[0] != tag[1]:
                    # print(tag)
                    right_match = True
            return right_match and left_match

        while True:
            if num_centers == 0 or index >= len(center_tags):
                break
            if index == 0:
                event_pieces.append([center_tags[index]])
                index += 1
            else:
                # pdb.set_trace()
                temp_event_pieces = []
                single_node = True
                for event_piece in event_pieces:
                    connected_centers = []
                    match_all = True
                    for center_tag_ in event_piece:
                        if center_match(center_tag_, center_tags[index]):
                            connected_centers.append(center_tag_)
                            single_node = False
                        else:
                            match_all = False
                    if match_all:
                        # pdb.set_trace()
                        event_piece.append(center_tags[index])
                    else:
                        if len(connected_centers) > 0:
                            temp_event_pieces.append(connected_centers + [center_tags[index]])
                event_pieces += temp_event_pieces
                if single_node:
                    event_pieces.append([center_tags[index]])
                index += 1

        deleted_events = []
        for i in range(len(event_pieces)):
            for j in range(i + 1, len(event_pieces)):
                if len(set(event_pieces[i]) - set(event_pieces[j])) == 0:
                    deleted_events.append(i)
                elif len(set(event_pieces[j]) - set(event_pieces[i])) == 0:
                    deleted_events.append(j)

        event_pieces_ = []
        for i in range(len(event_pieces)):
            if i in deleted_events:
                continue
            else:
                event_piece_list_ = []
                for event_piece in event_pieces[i]:
                    event_piece_list_.append((event_piece[0], self.id2tag[event_piece[2]]))
                event_pieces_.append(event_piece_list_)
        event_pieces = event_pieces_
        # pdb.set_trace()
        # event_piece = [offset, tag]
        # event_type_set = set([tag.split(self.separator)[0] for tag in self.tag2id])
        # tags_ = [[] for _ in range(predicted_matrix_tag.shape[0])]
        # for tag in tags:
        #     tags_[tag[0]].append([tag[1],tag[2]])
        # event_pieces = []
        event_list = []
        # for i in range(len(tags_)):
        #     tags_considered = [False] * len(tags_[i])
        #     if len(tags_considered) == 0:
        #         continue
        #     temp_event_pieces = []
        #     for event_piece_list in event_pieces:
        #         all_connect = True
        #         event_type = event_piece_list[0][1].split(self.separator)[0]
        #         connected_pieces = []
        #         for event_piece in event_piece_list:
        #             connect = False
        #             # pdb.set_trace()
        #             for j in range(len(tags_[i])):
        #                 tag = tags_[i][j]
        #                 if tag[0] > i:
        #                     continue
        #                 if tag[1] == event_piece[1] and tag[0] == event_piece[0]:
        #                     connect = True
        #                     connected_pieces.append(event_piece)
        #                     tags_considered[j] = True
        #                     break
        #             # pdb.set_trace()
        #             if not connect:
        #                 all_connect = False
        #         if all_connect:
        #             # event_piece_list = []
        #             for j in range(len(tags_[i])):
        #                 piece = tags_[i][j]
        #                 if piece[0] != i:
        #                     continue
        #                 match_all = True
        #                 for event_piece in event_piece_list:
        #                     offset = event_piece[0]
        #                     match = False
        #                     for piece_ in tags_[offset]:
        #                         if piece[0] == piece_[0] and piece[1] == piece_[1]:
        #                             match = True
        #                     if not match:
        #                         match_all = False
        #                 # pdb.set_trace()
        #                 if match_all:
        #                     event_piece_list.append([i, piece[1]])
        #                     tags_considered[j] = True
        #
        #         else:
        #             if len(connected_pieces) > 0:
        #                 has_B = False
        #                 for j in range(len(tags_[i])):
        #                     piece = tags_[i][j]
        #                     if piece[0] != i:
        #                         continue
        #                     if piece[1].split(self.separator)[0] == event_type and piece[0] <= i:
        #                         if 'B' == piece[1][-1]:
        #                             has_B = True
        #
        #                         match_all = True
        #                         for event_piece in connected_pieces:
        #                             offset = event_piece[0]
        #                             match = False
        #                             for piece_ in tags_[offset]:
        #                                 if piece[0] == piece_[0] and piece[1] == piece_[1]:
        #                                     match = True
        #                             if not match:
        #                                 match_all = False
        #                         # pdb.set_trace()
        #                         if match_all:
        #                             connected_pieces.append([i, piece[1]])
        #                             tags_considered[j] = True
        #
        #
        #
        #                 if has_B:
        #                     # pdb.set_trace()
        #                     temp_event_pieces.append(connected_pieces)
        #
        #     event_pieces += temp_event_pieces
        #
        #     for event_type in event_type_set:
        #         connected_pieces = []
        #         for j in range(len(tags_considered)):
        #             if tags_considered[j]:
        #                 continue
        #             tag = tags_[i][j]
        #             if tag[0] == i and tag[1].split(self.separator)[0] == event_type:
        #                 connected_pieces.append([i, tag[1]])
        #         if len(connected_pieces) > 0:
        #             event_pieces.append(connected_pieces)

        for event_piece_list in event_pieces:
            event = {
            }
            event['argument_list'] = []
            for i in range(len(event_piece_list)):
                event_piece = event_piece_list[i]
                tag = event_piece[1]
                tag_type = tag[:-2]  # wyc 1208
                event_type, role = tag_type.split(self.separator)  # wyc 1208
                if tag[-1] == 'B':
                    end_piece = event_piece
                    if 'Trigger' in tag:
                        for j in range(i + 1, len(event_piece_list)):
                            if 'Trigger' in event_piece_list[j][1]:
                                if event_piece_list[j][1][-1] == 'I' and tag_type in event_piece_list[j][1]:
                                    end_piece = event_piece_list[j]
                                else:
                                    break
                        event['trigger'] = ' '.join(text.split()[event_piece[0]: end_piece[0] + 1])
                        event['trigger_tok_span'] = [event_piece[0], end_piece[0] + 1]
                        event['trigger_char_span'] = [tok2char_span[event_piece[0]][0], tok2char_span[end_piece[0]][1]]
                        event['trigger_type'] = event_type  # wyc 1208
                    else:
                        for j in range(i + 1, len(event_piece_list)):
                            if event_piece_list[j][1][-1] == 'I' and tag_type in event_piece_list[j][1]:
                                end_piece = event_piece_list[j]
                            elif (end_piece[0] + 1) < event_piece_list[j][0]:
                                break

                        event['argument_list'].append({'text': ' '.join(text.split()[event_piece[0]:end_piece[0] + 1]),
                                                       "event_type": event_type,  # wyc 1208
                                                       'tok_span': [event_piece[0], end_piece[0] + 1],
                                                       'char_span': [tok2char_span[event_piece[0]][0],
                                                                     tok2char_span[end_piece[0]][1]],
                                                       'type': role,  # wyc 1208
                                                       })
            # try:
            # assert 'trigger' in event
            # except:
            #     pdb.set_trace()
            #     pass
            event_list.append(event)  # predicted event list

        pred_sample = copy.deepcopy(sample)
        # change to the predicted one, if not, will use ground truth to score
        pred_sample["event_list"] = event_list
        # pdb.set_trace()
        return pred_sample



