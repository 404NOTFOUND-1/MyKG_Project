import json
import re

import jieba


def chinese_word_cut(mytext):
    jieba.load_userdict('自定义词典.txt')  # 这里你可以添加jieba库识别不了的网络新词，避免将一些新词拆开
    jieba.initialize()  # 初始化jieba
    # 文本预处理 ：去除一些无用的字符只提取出中文出来
    new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
    new_data = " ".join(new_data)
    # 文本分词
    seg_list_exact = jieba.cut(new_data)
    result_list = []
    # 读取停用词库
    with open('停用词库.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()
        for i in con:
            i = i.replace("\n", "")  # 去掉读取每一行数据的\n
            stop_words.add(i)
    # 去除停用词并且去除单字
    for word in seg_list_exact:
        if word not in stop_words and len(word) > 1:
            result_list.append(word)
    return result_list


content = []
with open('all.jsonl', 'r', encoding='utf8') as f:
    for line in f:
        data = json.loads(line)
        content.append(data['text'])
# 保存分词结果
save_path = '分词后的数据.txt'  # 要保存的分词文件的路径
with open(save_path, 'a', encoding='utf-8') as f:
    for comment in content:
        cutted_comment = ' '.join(chinese_word_cut(comment))
        if cutted_comment:
            f.write(cutted_comment)
            f.write('\n')
