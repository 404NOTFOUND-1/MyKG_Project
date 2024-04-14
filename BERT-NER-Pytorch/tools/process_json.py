import json
import random

raw_file = 'doccano/新闻/all.jsonl'
conv_file = 'doccano/doccano_conv.json'


def output_json(file, data):
    """
    从列表输出json格式文件
    :param file: 输出文件路径（文件）
    :param data: 输入数据（列表）
    :return:
    """
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def train_test_output(data, type='train'):
    if type == 'train':
        output_json('cluener/train.json', data)
    elif type == 'dev':
        output_json('cluener/dev.json', data)
    elif type == 'test':
        output_json('cluener/rawtest.json', data)
        with open('cluener/rawtest.json', 'r', encoding="utf8") as f:
            with open('cluener/test.json', 'w', encoding="utf8") as fw:
                i = 0
                for line in f:
                    data = json.loads(line)
                    conv = {'id': i, 'text': data['text']}
                    json.dump(conv, fw, ensure_ascii=False)
                    fw.write('\n')
                    i += 1
            fw.close()
        f.close()


def jsonl2json(input_json):
    """
    将jsonl格式转换为cluener数据集格式
    :param input_json: jsonl格式
    :return: json格式
    """
    output_json = {'text': input_json['text'], 'label': {}}

    for entity in input_json['entities']:
        label = entity['label']
        start_offset = entity['start_offset']
        end_offset = entity['end_offset']
        entity_text = input_json['text'][start_offset:end_offset]
        if label not in output_json['label']:
            output_json['label'][label] = {}
        if entity_text not in output_json['label'][label]:
            output_json['label'][label][entity_text] = []
        output_json['label'][label][entity_text].append([start_offset, end_offset - 1])
    return output_json


def doccano2json(path):
    """
    从doccano输出转换为cluener数据集格式的json文件
    :param path: jsonl文件路径
    :return:
    """
    with open(path, 'r', encoding="utf-8") as f:
        with open(conv_file, 'w', encoding="utf-8") as fw:
            for line in f:
                data = json.loads(line)
                # print(data)
                conv_data = jsonl2json(data)
                # print(conv_data)
                json.dump(conv_data, fw, ensure_ascii=False)
                fw.write('\n')
        fw.close()
    f.close()
    print('jsonl转json文件完成！')


def train_dev_test_split(path, train_size, dev_size):
    """
    生成训练集、验证集、测试集（训练集和验证集的格式相同，测试集没有标注）
    :param path: 转换后的json文件路径
    :param train_size: 训练集占总数据的比例
    :param dev_size: 验证集占总数据的比例
    :return:
    """
    with open(path, 'r', encoding="utf8") as f:
        data_list = []
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    # 计算数据集划分的索引
    data_length = len(data_list)
    train_split = int(train_size * data_length)
    dev_split = int(dev_size * data_length)
    # 打乱数据顺序
    random.shuffle(data_list)
    # 划分训练集和验证集
    train_data = data_list[:train_split]
    dev_data = data_list[train_split:train_split + dev_split]
    test_data = data_list[train_split + dev_split:]
    # 输出训练集和验证集的长度
    print("训练集长度:", len(train_data))
    print("验证集长度:", len(dev_data))
    print("测试集长度:", len(test_data))
    # 输出训练集
    train_test_output(train_data, 'train')
    # 输出验证集
    train_test_output(dev_data, 'dev')
    # 输出测试集
    train_test_output(test_data, 'test')


def generate_train_dev_test_datasets(raw_jsonl, train_size=0.8, dev_size=0.1):
    """
    直接调用，将原始jsonl文件转换为训练集、验证集和测试集
    :param raw_jsonl: 原始jsonl
    :param train_size: 训练集占数据集的比例
    :param dev_size: 验证集占数据集的比例
    :return:
    """
    doccano2json(raw_jsonl)
    train_dev_test_split(conv_file, train_size, dev_size)


def videotxt_wrap_horizontal(path, width):
    """
    视频获取的json文本转换为可以标注的textline形式
    :param path: 原始文本
    :param width: 目标文本每一行的长度
    :return:
    """
    lines = []
    current_line = ""
    input_text = ""
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            input_text += line + '\n'
    words = input_text.split()

    for word in words:
        if len(current_line) + len(word) <= width:
            current_line += word + "，"
        else:
            lines.append(current_line.strip())
            current_line = word + "，"
    if current_line:
        lines.append(current_line.strip())
    return lines


def video2text(src, width, show=False):
    """
    从bilibili视频字幕中获得文本用于标注
    :param src: 源json路径
    :param width: 清洗后文本的最大宽度
    :param show: 显示清洗后的文本（默认不显示）
    :return:
    """
    temp_file = src + 'temp.json'
    temp = json.load(open(temp_file, encoding='gbk'))
    raw_file = src + 'rawtext.txt'
    with open(raw_file, 'w', encoding='utf8') as f:
        for content in temp['body']:
            output_txt = content['content']
            f.write(output_txt + '\n')
    f.close()
    print('json转rawtext完成')
    final_file = src + 'text.txt'
    wrapped_lines = videotxt_wrap_horizontal(raw_file, width)
    with open(final_file, 'a', encoding='utf8') as f:
        for line in wrapped_lines:
            if show:
                print(line)
            f.write(line + '\n')
    f.close()
    print('标注数据清洗完成！')


def generate_ner_datasets(src):
    """
    从获取的原始文本中直接得到用于命名实体识别的数据集
    :param if_video:
    :param width:
    :param src:
    :return:
    """
    with open(src + 'text.txt', 'r', encoding='utf8') as f:
        with open('cluener/test.json', 'w', encoding='utf8') as fw:
            i = 0
            for line in f:
                conv = {'id': i, 'text': line.replace('\n', '')}
                json.dump(conv, fw, ensure_ascii=False)
                fw.write('\n')
                i += 1
        fw.close()
    f.close()
    print('命名实体数据集生成完成')


if __name__ == '__main__':
    generate_train_dev_test_datasets(raw_file)
