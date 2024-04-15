import json

from tqdm import tqdm


def get_entity(dat):
    """
    获取实体
    :param dat:
    :return:
    """
    text = dat['text']
    targets = []
    for eles in dat['entities']:
        target = {"id": eles['id'], "type": {}, "name": {}}
        # print(eles['id'], eles['label'], eles['start_offset'], eles['end_offset'])
        start = eles['start_offset']
        end = eles['end_offset']
        target['type'] = eles['label']
        target['name'] = text[start:end]
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
    for i, Relation in enumerate(Relations):
        spo = {"predicate": {}, "object_type": {}, "subject_type": {}, "object": {}, "subject": {}}
        spo['predicate'] = Relation['type']
        from_id = Relation['from_id']
        to_id = Relation['to_id']
        for part in Parts:
            if part['id'] == from_id:
                spo['subject_type'] = part['type']
                spo['subject'] = part['name']
            if part['id'] == to_id:
                spo['object_type'] = part['type']
                spo['object'] = part['name']
        # print(spo)
        spo_list.append(spo)
    return spo_list


def gen_spo_list(output='output.json'):
    """
    输出最终的三元组列表
    :param output: 输出名称
    :return:
    """
    text_spo = {"text": {}, "spo_list": {}}
    with open('all.jsonl', 'r', encoding='utf8') as f:
        with open(output, 'w') as fw:
            for line in tqdm(f, 'doccano to json'):
                data = json.loads(line)
                if data['relations']:
                    text, parts = get_entity(data)
                    spo_lists = get_relation(parts, data['relations'])
                    text_spo['text'] = text
                    text_spo['spo_list'] = spo_lists
                    # print(text_spo)
                    fw.write(json.dumps(text_spo, ensure_ascii=False))
                    fw.write('\n')
