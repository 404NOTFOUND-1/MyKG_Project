# -*- coding: utf-8 -*-
import codecs
import json
import random

from tqdm import tqdm


class Doccano_preprocess:

    def __init__(self):
        self.input = './data/dgre/ori_data/all.jsonl'
        self.output = './data/dgre/ori_data/output.json'
        self.is_jsonl = True

    def gen_spo_list(self):
        """
        输出最终的三元组列表
        :return:
        """

        def get_entity(dat):
            """
            获取实体
            :param dat:
            :return:
            """
            text = dat['text']
            targets = []
            for eles in dat['entities']:
                target = {"id": eles['id'], "type": {}, "name": {}, "pos": {}}
                # print(eles['id'], eles['label'], eles['start_offset'], eles['end_offset'])
                start = eles['start_offset']
                end = eles['end_offset']
                target['type'] = eles['label']
                target['name'] = text[start:end]
                target['pos'] = [start, end]
                targets.append(target)
            return text, targets

        def get_relation(Parts, Relations):
            """
            获取关系
            :param Parts: 实体列表
            :param Relations: 关系列表
            :return:
            """
            spo_list = []
            if Relations:
                for i, Relation in enumerate(Relations):
                    spo = {"h": {"name": {}, "type": {}, "pos": {}}, "t": {"name": {}, "type": {}, "pos": {}},
                           'relation': Relation['type']}
                    from_id = Relation['from_id']
                    to_id = Relation['to_id']
                    for part in Parts:
                        if part['id'] == from_id:
                            # spo['subject_type'] = part['type']
                            spo['h']['name'] = part['name']
                            spo['h']['type'] = part['type']
                            spo['h']['pos'] = part['pos']
                        if part['id'] == to_id:
                            # spo['object_type'] = part['type']
                            spo['t']['name'] = part['name']
                            spo['t']['type'] = part['type']
                            spo['t']['pos'] = part['pos']
                    print(spo)
                    spo_list.append(spo)
            else:
                spo = {"h": {"name": {}, "type": {}, "pos": {}}, "t": {"name": {}, "type": {}, "pos": {}},
                       'relation': '没关系'}
                part_num = len(Parts)
                for i in range(part_num):
                    for j in range(i + 1, part_num):
                        spo['h']['name'] = Parts[i]['name']
                        spo['h']['type'] = Parts[i]['type']
                        spo['h']['pos'] = Parts[i]['pos']
                        spo['t']['name'] = Parts[j]['name']
                        spo['t']['type'] = Parts[j]['type']
                        spo['t']['pos'] = Parts[j]['pos']
                        spo_list.append(spo)
                        if j > 0:
                            break
                    if i > 0:
                        break

            return spo_list

        text_spo = {"text": {}, "spo_list": {}}
        sum_line = 0
        if self.is_jsonl:
            for _ in open(self.input, 'r', encoding='utf8'):
                sum_line += 1
            with open(self.input, 'r', encoding='utf8') as f:
                with open(self.output, 'w', encoding='utf8') as fw:
                    pbar = tqdm(total=sum_line, desc='doccano to json')
                    for line in f:
                        data = json.loads(line)
                        # if data['relations']:
                        text, parts = get_entity(data)
                        spo_lists = get_relation(parts, data['relations'])
                        text_spo['text'] = text
                        text_spo['spo_list'] = spo_lists
                        # print(text_spo)
                        fw.write(json.dumps(text_spo, ensure_ascii=False))
                        fw.write('\n')
                        pbar.update(1)
        else:
            with open(self.input, 'r') as f:
                with open(self.output, 'w') as fw:
                    for line in f:
                        fw.write(line)


class ProcessDgreData:
    def __init__(self):
        self.data_path = "./data/dgre/"
        self.train_file = self.data_path + "ori_data/output.json"
        # 需要修改文件路径

    def get_ner_data(self):
        with codecs.open(self.train_file, 'r', encoding="utf-8", errors="replace") as fp:
            data = fp.readlines()
        res = []
        for did, d in enumerate(tqdm(data, desc='get_ner_data')):
            d = eval(d)
            tmp = {}
            text = d['text']
            # tmp["id"] = d['ID']
            tmp['text'] = [i for i in text]
            tmp["labels"] = ["O"] * len(tmp['text'])
            for rel_id, spo in enumerate(d['spo_list']):
                h = spo['h']
                t = spo['t']
                h_start = h["pos"][0]
                h_end = h["pos"][1]
                t_start = t["pos"][0]
                t_end = t["pos"][1]
                tmp["labels"][h_start] = "B-" + h['type']
                for i in range(h_start + 1, h_end):
                    tmp["labels"][i] = "I-" + h['type']
                tmp["labels"][t_start] = "B-" + t['type']
                for i in range(t_start + 1, t_end):
                    tmp["labels"][i] = "I-" + t['type']
            res.append(tmp)
        train_ratio = 0.65
        train_num = int(len(res) * train_ratio)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "ner_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "ner_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        # 这里标签一般从数据中处理得到，这里我们自定义
        labels = ["PAR", "TYPE", "MAT"]
        with open(self.data_path + "ner_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))

    def get_re_data(self):
        with codecs.open(self.train_file, 'r', encoding="utf-8", errors="replace") as fp:
            data = fp.readlines()
        res = []
        re_labels = set()
        for did, d in enumerate(tqdm(data, desc='get_re_data')):
            d = eval(d)
            text = d['text']
            bks = []  # 存储包括
            ljds = []  # 存储连接到
            sbj_obj = []  # 存储真实包括-连接到
            for rel_id, spo in enumerate(d['spo_list']):
                tmp = {}
                tmp['text'] = text
                tmp["labels"] = []
                h = spo['h']
                t = spo['t']
                h_name = h["name"]
                t_name = t["name"]
                relation = spo["relation"]
                tmp_rel_id = str(did) + "_" + str(rel_id)
                tmp["id"] = tmp_rel_id
                tmp["labels"] = [h_name, t_name, relation]
                re_labels.add(relation)
                res.append(tmp)
                if h_name not in bks:
                    bks.append(h_name)
                if t_name not in ljds:
                    ljds.append(t_name)
                sbj_obj.append((h_name, t_name))

            # 关键是怎么构造负样本
            # 如果不在sbj_obj里则视为没有关系
            tmp = {}
            tmp["text"] = text
            tmp["labels"] = []
            tmp["id"] = str(did) + "_" + "norel"
            if len(bks) > 1 and len(ljds) > 1:
                neg_total = 3
                neg_cur = 0
                for bk in bks:
                    random.shuffle(ljds)
                    # print(gzyys)
                    for ljd in enumerate(ljds):
                        if (bk, ljd[1]) not in sbj_obj:
                            print([bk, ljd[1], "没关系"])
                            tmp["labels"] = [bk, ljd[1], "没关系"]
                            res.append(tmp)
                            neg_cur += 1
                        break
                    if neg_cur == neg_total:
                        break

        train_ratio = 0.6
        train_num = int(len(res) * train_ratio)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "re_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "re_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        # 这里标签一般从数据中处理得到，这里我们自定义
        labels = list(re_labels)
        with open(self.data_path + "re_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))


# class ProcessDuieData:
#     def __init__(self):
#         self.data_path = "./data/duie/"
#         self.train_file = self.data_path + "ori_data/duie_train.json"
#         self.dev_file = self.data_path + "ori_data/duie_dev.json"
#         self.test_file = self.data_path + "ori_data/duie_test2.json"
#         self.schema_file = self.data_path + "ori_data/duie_schema.json"
#
#     def get_rels(self):
#         rels = set()
#
#         with open(self.schema_file, 'r', encoding="utf-8") as fp:
#             lines = fp.readlines()
#             for line in lines:
#                 data = eval(line)
#                 rels.add(data['predicate'])
#
#         with open(self.data_path + "re_data/labels.txt", 'w', encoding="utf-8") as fp:
#             fp.write("\n".join(["没关系"] + list(rels)))
#
#     def get_ents(self):
#         ents = set()
#         rels = defaultdict(list)
#         with open(self.schema_file, 'r', encoding="utf-8") as fp:
#             lines = fp.readlines()
#             for line in lines:
#                 data = eval(line)
#                 subject_type = data['subject_type']['@value'] if '@value' in data['subject_type'] else data[
#                     'subject_type']
#                 object_type = data['object_type']['@value'] if '@value' in data['object_type'] else data['object_type']
#                 if "人物" in subject_type:
#                     subject_type = "人物"
#                 if "人物" in object_type:
#                     object_type = "人物"
#                 ents.add(subject_type)
#                 ents.add(object_type)
#                 predicate = data["predicate"]
#                 rels[subject_type + "_" + object_type].append(predicate)
#
#         with open(self.data_path + "ner_data/labels.txt", "w", encoding="utf-8") as fp:
#             fp.write("\n".join(list(ents)))
#
#         with open(self.data_path + "re_data/rels.txt", "w", encoding="utf-8") as fp:
#             json.dump(rels, fp, ensure_ascii=False, indent=2)
#
#     def get_ner_data(self, input_file, output_file):
#         res = []
#         with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
#             lines = fp.read().strip().split("\n")
#             for i, line in enumerate(tqdm(lines)):
#                 try:
#                     line = eval(line)
#                 except Exception as e:
#                     continue
#                 tmp = {}
#                 text = line['text']
#                 tmp['text'] = [i for i in text]
#                 tmp["labels"] = ["O"] * len(text)
#                 tmp['id'] = i
#                 spo_list = line['spo_list']
#                 for j, spo in enumerate(spo_list):
#                     # 从句子里面找到实体的开始位置、结束位置
#                     if spo['subject'] == "" or spo['object']['@value'] == "":
#                         continue
#                     try:
#                         subject_re_res = re.finditer(re.escape(spo['subject']), line['text'])
#                         subject_type = spo["subject_type"]
#                         if "人物" in subject_type:
#                             subject_type = "人物"
#                     except Exception as e:
#                         print(e)
#                         print(spo['subject'].replace('+', '\+'), line['text'])
#                         import sys
#                         sys.exit(0)
#                     for sbj in subject_re_res:
#                         sbj_span = sbj.span()
#                         sbj_start = sbj_span[0]
#                         sbj_end = sbj_span[1]
#                         tmp["labels"][sbj_start] = f"B-{subject_type}"
#                         for j in range(sbj_start + 1, sbj_end):
#                             tmp["labels"][j] = f"I-{subject_type}"
#                     try:
#                         object_re_res = re.finditer(
#                             re.escape(spo['object']['@value']), line['text'])
#                         object_type = spo['object_type']['@value']
#                         if "人物" in object_type:
#                             object_type = "人物"
#                     except Exception as e:
#                         print(e)
#                         print(line)
#                         print(spo['object']['@value'].replace('+', '\+').replace('(', ''), line['text'])
#                         import sys
#                         sys.exit(0)
#                     for obj in object_re_res:
#                         obj_span = obj.span()
#                         obj_start = obj_span[0]
#                         obj_end = obj_span[1]
#                         tmp["labels"][obj_start] = f"B-{object_type}"
#                         for j in range(obj_start + 1, obj_end):
#                             tmp["labels"][j] = f"I-{object_type}"
#                 res.append(tmp)
#
#         with open(output_file, 'w', encoding="utf-8") as fp:
#             fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))
#
#     def get_re_data(self, input_file, output_file):
#         res = []
#
#         with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
#             lines = fp.read().strip().split("\n")
#             for i, line in enumerate(tqdm(lines)):
#                 try:
#                     line = eval(line)
#                 except Exception as e:
#                     continue
#                 tmp = {}
#                 text = line['text']
#                 tmp['text'] = text
#                 tmp['id'] = i
#                 spo_list = line['spo_list']
#
#                 ent_rel_dict = defaultdict(list)
#                 sub_obj = []  # 用于存储关系对
#                 for j, spo in enumerate(spo_list):
#                     if spo['subject'] == "" or spo['object']['@value'] == "":
#                         continue
#                     sbj = spo['subject']
#                     obj = spo['object']['@value']
#                     tmp["labels"] = [sbj, obj, spo["predicate"]]
#                     sub_obj.append((sbj, obj))
#                     ent_rel_dict[spo["predicate"]].append((sbj, obj))
#                     res.append(tmp)
#
#                 # 重点是怎么构造负样本：没有关系的
#                 for k, v in ent_rel_dict.items():
#                     sbjs = list(set([p[0] for p in v]))
#                     objs = list(set([p[1] for p in v]))
#                     if len(sbjs) > 1 and len(objs) > 1:
#                         neg_total = 3
#                         neg_cur = 0
#                         for sbj in sbjs:
#                             random.shuffle(objs)
#                             for obj in objs:
#                                 if (sbj, obj) not in sub_obj:
#                                     tmp["id"] = str(i) + "_" + "norel"
#                                     tmp["labels"] = [sbj, obj, "没关系"]
#                                     res.append(tmp)
#                                     neg_total += 1
#                                 break
#                             if neg_cur == neg_total:
#                                 break
#
#         with open(output_file, 'w') as fp:
#             fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))


if __name__ == "__main__":
    pre = Doccano_preprocess()
    pre.gen_spo_list()
    processDgreData = ProcessDgreData()

    processDgreData.get_ner_data()
    processDgreData.get_re_data()

    # processDuieData = ProcessDuieData()
    # processDuieData.get_ents()
    # processDuieData.get_rels()
    # processDuieData.get_ner_data(processDuieData.train_file,
    #                             os.path.join(processDuieData.data_path, "ner_data/train.txt"))
    # processDuieData.get_ner_data(processDuieData.dev_file, os.path.join(processDuieData.data_path, "ner_data/dev.txt"))
    # processDuieData.get_re_data(processDuieData.train_file,
    #                             os.path.join(processDuieData.data_path, "re_data/train.txt"))
    # processDuieData.get_re_data(processDuieData.dev_file, os.path.join(processDuieData.data_path, "re_data/dev.txt"))
