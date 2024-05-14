import json

from tqdm import tqdm

with open('data.json', 'r', encoding='utf8') as f:
    with open('output.json', 'w', encoding='utf8') as fw:
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
                print(output)
                json.dump(output, fw, ensure_ascii=False)
                fw.write('\n')
