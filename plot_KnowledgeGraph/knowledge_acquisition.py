import argparse

import networkx as nx
from neo4j import GraphDatabase
from pyvis.network import Network


# Neo4j数据库连接类
class Neo4jConnection:
    def __init__(self, url, user, password):
        self._driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query, parameters=None, db=None):
        session = None
        response = None
        try:
            session = self._driver.session(database=db) if db is not None else self._driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


# 查询Neo4j数据库并获取结果
def fetch_data(uri, user, password, cypher_query):
    conn = Neo4jConnection(uri, user, password)
    results = conn.query(cypher_query)
    conn.close()
    return results


# 将查询结果转换为关系图
def build_graph(args, results):
    G = nx.DiGraph()  # 使用有向图
    if args.query_type == 'type1' or args.query_type == 'type2':
        for record in results:
            node1 = record['n1']['name']
            node2 = record['n2']['name']
            relationship = record['r'].type
            # 使用 get 方法获取节点的第一个标签，否则默认 'Unknown'
            node1_label = list(record['n1'].labels)[0] if record['n1'].labels else 'Unknown'
            node2_label = list(record['n2'].labels)[0] if record['n2'].labels else 'Unknown'
            G.add_node(node1, label=node1_label)
            G.add_node(node2, label=node2_label)
            G.add_edge(node1, node2, label=relationship)
    else:
        for record in results:
            node = record['n']['name']
            node_label = list(record['n'].labels)[0] if record['n'].labels else 'Unknown'
            G.add_node(node, label=node_label)
    return G


# 分配节点颜色
def get_node_color(label):
    color_map = {
        'SYS': 'deeppink',
        'METHOD': 'seagreen',
        'TYPE': 'orange',
        'MAT': 'aqua',
        'PAR': 'orchid',
        # 添加更多标签和颜色的映射
    }
    return color_map.get(label)  # 默认颜色为灰色


# 打印节点标签信息
def print_labels(args, results):
    if args.query_type == 'type1' or args.query_type == 'type2':
        for record in results:
            node1_labels = record['n1'].labels
            node1_name = record['n1'].get('name')
            node2_labels = record['n2'].labels
            node2_name = record['n2'].get('name')
            print(f"Node 1 : name:{node1_name}  label:{node1_labels}")
            print(f"Node 2 : name:{node2_name}  label:{node2_labels}")
            print()
    else:
        for record in results:
            node_name = record['n'].get('name')
            node_labels = record['n'].labels
            print(f"Node : name:{node_name}  label:{node_labels}")
            print()


# 使用Pyvis可视化关系图
def visualize_graph_pyvis(G):
    net = Network(notebook=True)
    for node, data in G.nodes(data=True):
        color = get_node_color(data['label'])
        net.add_node(node, label=node, title=node, color=color)
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]['label'])
    net.show("graph.html")


def generate_query(query_type, keyword1, keyword2, keyword3):
    query_templates = {
        "type1": f"MATCH (n1:{keyword1})-[r:{keyword2}]->(n2:{keyword3}) RETURN n1, n2, r",
        "type2": f"MATCH (n1)-[r:{keyword2}]->(n2) RETURN n1, n2, r",
        "type3": f"MATCH (n:{keyword1}) RETURN n",
        # 添加更多模板
    }
    print(query_templates[query_type])
    return query_templates.get(query_type, "")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Query Neo4j and visualize the graph.")
    parser.add_argument('--query_type', type=str, required=True, help="Type of query to execute.")
    parser.add_argument('--keyword1', type=str, required=True, help="Keyword for the query.")
    parser.add_argument('--keyword2', type=str, required=True, help="Keyword for the query.")
    parser.add_argument('--keyword3', type=str, required=True, help="Keyword for the query.")
    args = parser.parse_args()

    # 生成查询语句
    cypher_query = generate_query(args.query_type, args.keyword1, args.keyword2, args.keyword3)
    if not cypher_query:
        print("Invalid query type.")
        return
    # 示例连接信息
    uri = "bolt://localhost:7687"  # 替换为你的Neo4j服务器URI
    user = "neo4j"  # 替换为你的Neo4j用户名
    password = "12345678"  # 替换为你的Neo4j密码
    # 执行查询并获取结果
    results = fetch_data(uri, user, password, cypher_query)
    # 打印节点标签信息
    print_labels(args, results)
    # 构建关系图
    G = build_graph(args, results)
    # 可视化关系图
    visualize_graph_pyvis(G)


if __name__ == "__main__":
    main()
