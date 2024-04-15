from py2neo import Graph, NodeMatcher
from py2neo.data import Node, Relationship

from doccano_preprocess import *

# spo_list = [{"predicate": "作者", "object_type": "人物", "subject_type": "图书作品", "object": "余国藩",
#              "subject": "红楼梦、西游记与其他"},
#             {"predicate": "出版社", "object_type": "出版社", "subject_type": "图书作品",
#              "object": "生活·读书·新知三联书店", "subject": "红楼梦、西游记与其他"},
#             {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "三世森情",
#              "subject": "烈女子"},
#             {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "丁克森",
#              "subject": "烈女子"}]

if __name__ == '__main__':
    spo_path = 'output.json'
    gen_spo_list(spo_path)

    spo_list = []
    with open(spo_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            spo_list.extend(data['spo_list'])

    my_graph = Graph("http://localhost:7474/",
                     name="neo4j", password="12345678")
    my_graph.delete_all()
    my_graph.create(Node(spo_list[0]['subject_type'], name=spo_list[0]['subject']))
    my_graph.create(Node(spo_list[0]['object_type'], name=spo_list[0]['object']))
    matcher = NodeMatcher(my_graph)
    # test2
    for i, spo in tqdm(enumerate(spo_list), '构建图谱中'):
        predicate = spo['predicate']
        object_type = spo['object_type']
        subject_type = spo['subject_type']
        object_name = spo['object']
        subject_name = spo['subject']
        nodelist1 = list(matcher.match(subject_type, name=subject_name))
        if len(nodelist1) > 0:
            nodelist2 = list(matcher.match(object_type, name=object_name))
            if len(nodelist2) > 0:
                zhucong = Relationship(nodelist1[0], predicate, nodelist2[0])
                my_graph.create(zhucong)
            else:
                cong = Node(object_type, name=object_name)
                my_graph.create(cong)
                zhucong = Relationship(nodelist1[0], predicate, cong)
                my_graph.create(zhucong)
        else:
            zhu = Node(subject_type, name=subject_name)
            my_graph.create(zhu)
            nodelist2 = list(matcher.match(object_type, name=object_name))
            if len(nodelist2) > 0:
                zhucong = Relationship(zhu, predicate, nodelist2[0])
                my_graph.create(zhucong)
            else:
                cong = Node(object_type, name=object_name)
                my_graph.create(cong)
                zhucong = Relationship(zhu, predicate, cong)
                my_graph.create(zhucong)
    print('图谱生成完成!')
