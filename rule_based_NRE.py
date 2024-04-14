from jieba import posseg

allowRelationships = ['母亲', '父亲', '儿子', '女儿', '母', '父', '下嫁', '又嫁', '祖父', '祖母', '孙', '孙子', '改嫁',
                      '哥哥', '姐姐', '弟弟']
content = ['武德九年(626年)，玄武门之变后，李渊退位称太上皇，禅位于儿子李世民',
           '李渊的祖父李虎，在西魏时官至太尉，是西魏八柱国之一。李渊的父亲李昞，北周时历官御史大夫、安州总管、柱国大将军，袭封唐国公。李渊的母亲是隋文帝独孤皇后的姐姐[7]。']
# 抽取的实体词性
allowTags = ['nr']
relationships = set()
for line in content:
    sentence = []
    for word, tag in posseg.cut(line):
        if tag == 'nr' or word in allowRelationships:
            sentence.append(word)
    sentence = ' '.join(sentence)
    sentenceSplit = sentence.split(' ')
    print("*** 原始文本 ***\n", line)
    for i in range(1, len(sentenceSplit) - 1):
        if sentenceSplit[i] in allowRelationships:
            source = sentenceSplit[i - 1]
            relationship = sentenceSplit[i]
            target = sentenceSplit[i + 1]
            # 不同场景中需要根据实际情况进行一些特殊的过滤
            print('提取结果：', source + '->' + relationship + '->' + target)
            relationships.add(source + '->' + relationship + '->' + target)
