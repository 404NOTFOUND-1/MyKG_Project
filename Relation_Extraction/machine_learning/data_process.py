import json


def svm_data_process():
    with open('data.json', 'r', encoding='utf8') as f:
        with open('svm_output.json', 'w', encoding='utf8') as fw:
            output = {'text': {}, 'entities': {}, 'relations': {}}
            for line in f:
                data = json.loads(line)
                for spo in data['spo_list']:
                    # print(spo['h']['name'], spo['t']['name'], spo['relation'])
                    if spo['relation'] != '没关系':
                        text = spo['h']['name'] + spo['relation'] + spo['t']['name']
                    else:
                        text = spo['h']['name'] + '与' + spo['t']['name'] + spo['relation']
                    output['text'] = text
                    output['entities'] = (spo['h']['name'], spo['t']['name'])
                    output['relations'] = spo['relation']
                    # print(output)
                    json.dump(output, fw, ensure_ascii=False)
                    fw.write('\n')
            print('svm_output.json done')


##=======================================##

##############################################
##############################################
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


def doccano2json(path, conv_file):
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


def bilstm_data_process(raw_jsonl, conv_file):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    doccano2json(raw_jsonl, conv_file)
    json_data = []
    with open(conv_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    # json_data中每一条数据的格式为
    '''
    {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
     'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
     '''

    # 将json文件处理成如下格式
    '''
    [['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另', '一', 
    '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老', '桂', '认', 
    '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'], 
    ['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 
    'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    '''
    data = []
    # 遍历json_data中每组数据
    for i in range(len(json_data)):
        # 将标签全初始化为'O'
        label = ['O'] * len(json_data[i]['text'])
        # 遍历'label'中几组实体，如样例中'name'和'company'
        for n in json_data[i]['label']:
            # 遍历实体中几组文本，如样例中'name'下的'叶老桂'（有多组文本的情况，样例中只有一组）
            for key in json_data[i]['label'][n]:
                # 遍历文本中几组下标，如样例中[[9, 11]]（有时某个文本在该段中出现两次，则会有两组下标）
                for n_list in range(len(json_data[i]['label'][n][key])):
                    # 记录实体开始下标和结尾下标
                    start = json_data[i]['label'][n][key][n_list][0]
                    end = json_data[i]['label'][n][key][n_list][1]
                    # 将开始下标标签设为'B-' + n，如'B-' + 'name'即'B-name'
                    # 其余下标标签设为'I-' + n
                    label[start] = 'B-' + n
                    label[start + 1: end + 1] = ['I-' + n] * (end - start)

        # 对字符串进行字符级分割
        # 英文文本如'bag'分割成'b'，'a'，'g'三位字符，数字文本如'125'分割成'1'，'2'，'5'三位字符
        texts = []
        for t in json_data[i]['text']:
            texts.append(t)
        # 将文本和标签编成一个列表添加到返回数据中
        data.append([texts, label])
    return data
