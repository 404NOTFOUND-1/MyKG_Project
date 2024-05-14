import json
import os
from collections import namedtuple

import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer

from model import BertNer, BertRe


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.re_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "re_args.json"), "re_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.re_id2label = {int(k): v for k, v in self.re_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        print(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin"))
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.re_model = BertRe(self.re_args)
        print(os.path.join(self.re_args.output_dir, "pytorch_model_re.bin"))
        self.re_model.load_state_dict(torch.load(os.path.join(self.re_args.output_dir, "pytorch_model_re.bin")))
        self.re_model.to(self.device)
        if os.path.exists(os.path.join(self.re_args.data_path, "rels.txt")):
            with open(os.path.join(self.re_args.data_path, "rels.txt"), "r") as fp:
                self.rels = json.load(fp)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def re_tokenizer(self, text, h, t):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        pre_length = 4 + len(h) + len(t)
        text = text[:self.max_seq_len - pre_length]
        text = "[CLS]" + h + "[SEP]" + t + "[SEP]" + text + "[SEP]"
        tmp_input_ids = self.tokenizer.tokenize(text)
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        token_type_ids = [0] * self.max_seq_len
        input_ids = torch.tensor(np.array([input_ids]))
        token_type_ids = torch.tensor(np.array([token_type_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask, token_type_ids

    def re_predict_common(self, hs, ts):
        res = []
        tmp = []
        # 用于标识h和next_h之间是否有t
        flag = False
        next_h = None
        for i, h in enumerate(hs):
            if i + 1 < len(hs):
                next_h = hs[i + 1]
            for t in ts:
                h_start = h[1]
                h_end = h[2]
                t_start = t[1]
                t_end = t[2]
                # =============================================
                # 定义不同数据的后处理规则
                if self.data_name == "dgre":
                    # 该数据原因不会出现在设备前面
                    if t_end < h_start:
                        continue
                    if next_h and h_start < t_start < next_h[1]:
                        flag = True
                    # 如果两个设备之间有原因，当前原因在第二个设备之后
                    # 那么第一个设备的原因就不能是它，于是结束原因循环
                    if next_h and flag and t_start > next_h[2]:
                        flag = False
                        break
                elif self.data_name == "duie":
                    if h[0] == t[0]:
                        continue
                # =============================================
                if (h[0], t[0]) in tmp:
                    continue
                tmp.append((h[0], t[1]))
                input_ids, attention_mask, token_type_ids = self.re_tokenizer(text, h[0], t[0])
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                output = self.re_model(input_ids, token_type_ids, attention_mask)
                logits = output.logits
                score = torch.softmax(logits, dim=1)
                score = score.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                logits = np.argmax(logits, -1)
                score = score[0][logits[0]]
                rel = self.re_id2label[logits[0]]
                if rel != "没关系" and (h[0], t[0], rel) not in res:
                    res.append((h[0], t[0], rel))
        return res

    def re_predict_dgre(self, text, ner_result):
        try:
            hs = ner_result["PAR"]
            ts = ner_result["PAR"]
            res = self.re_predict_common(hs, ts)
        except Exception as e:
            res = []
        return res

    def re_predict_duie(self, text, ner_result):
        result = []
        for k, v in self.rels.items():
            ent = k.split("_")
            ent1 = ent[0]
            ent2 = ent[1]
            if ent1 in ner_result and ent2 in ner_result:
                hs = ner_result[ent1]
                ts = ner_result[ent2]
                res = self.re_predict_common(hs, ts)
                result.extend(res)
        return res

    def re_predict(self, text, ner_result):
        res = []
        if self.data_name == "dgre":
            res = self.re_predict_dgre(text, ner_result)
        elif self.data_name == "duie":
            res = self.re_predict_duie(text, ner_result)
        return res

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "dgre"
    predictor = Predictor(data_name)
    if data_name == "dgre":
        texts = [
            "根据单体电池封装形式不同，可以分为圆柱电芯、方形电芯及软包电芯。",
            "电池单体模块将一定数量的单体电池连接在一起。",
            "电池单体模块放在一个框架中组成电池包的组件。",
            "盖子打开后，互锁开关断开，三个互锁开关串联与ecu连接，任何一个互锁开关断开时，ecu就会断开主继电器，",
            "电池管理系统包括了电池管理器、熔断器、手动维修开关（MSD）、接触器、霍尔传感器/分流器、绝缘模块",
        ]
    elif data_name == "duie":
        texts = [
            "歌曲《墨写你的美》是由歌手冷漠演唱的一首歌曲",
            "982年，阎维文回到山西，隆重地迎娶了刘卫星",
            "王皃姁为还是太子的刘启生了二个儿子，刘越（汉景帝第11子）、刘寄（汉景帝第12子）",
            "数据分析方法五种》是2011年格致出版社出版的图书，作者是尤恩·苏尔李",
            "视剧《不可磨灭》是导演潘培成执导，刘蓓、丁志诚、李洪涛、丁海峰、雷娟、刘赫男等联袂主演",
        ]
    for text in texts:
        ner_result = predictor.ner_predict(text)
        print("文本>>>>>：", text)
        print("实体>>>>>：", ner_result)
        re_result = predictor.re_predict(text, ner_result)
        print("关系>>>>>：", re_result)
        print("=" * 100)
