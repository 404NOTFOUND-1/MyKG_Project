import argparse

from process_json import *

name = '动力电池'
raw_file = 'doccano/{}/all.jsonl'.format(name)
temp = 'doccano/temp/'
conv_file = 'doccano/动力电池/doccano_conv.json'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=str, default='doccano/temp/', help='temp文件夹目录')
    parser.add_argument('--split_raw_jsonl', type=str, default='doccano/动力电池/all.jsonl',
                        help='用于分割的原始jsonl数据集')
    parser.add_argument('--do_split', action='store_true', help='用于选择是否分割数据集')
    parser.add_argument('--do_annotate', action='store_true', help='用于选择是否生成标注数据集')
    parser.add_argument('--do_gen_ner', action='store_true', help='用于选择直接从给定文本生成测试集')
    return parser


if __name__ == '__main__':
    arg = get_parser().parse_args()
    # 从视频字幕生成标注数据集
    if arg.do_annotate:
        video2text(arg.temp, 60)
    # 分割标注好的数据集为训练集、验证集、测试集
    if arg.do_split:
        print('划分文件目录：{}'.format(arg.split_raw_jsonl))
        generate_train_dev_test_datasets(arg.split_raw_jsonl, conv_file, train_size=0.45, dev_size=0.4)
    if arg.do_gen_ner:
        print('生成实体检测数据集')
        generate_ner_datasets(arg.temp)
