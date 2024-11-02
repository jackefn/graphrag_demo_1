from neo4j import GraphDatabase

# 定义连接参数
uri = "neo4j://localhost:7687"  # Neo4j 连接 URI
username = "neo4j"             # Neo4j 用户名
password = "mice95167TB"       # Neo4j 密码

# 创建驱动对象
driver = GraphDatabase.driver(uri, auth=(username, password))

# 定义一个测试连接的函数
def test_connection(driver):
    with driver.session() as session:
        # 运行一个简单的查询来测试连接
        result = session.run("RETURN 'Connected to Neo4j!' AS message")
        for record in result:
            print(record["message"])

# 调用测试连接的函数
test_connection(driver)

# 定义一个函数来导入数据到 Neo4j
def import_data(driver):
    with driver.session() as session:
        # 导入节点，并使用节点类型作为标签
        session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///long_nodes.csv' AS row
            CALL apoc.create.node([row.`节点类型`], {name: row.`节点名称`}) YIELD node
            RETURN node;
        """)

        # 导入边，并使用合并后的边类型作为关系类型
        session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///long_edges.csv' AS row
            MATCH (n1 {name: row.`源节点`})
            MATCH (n2 {name: row.`目标节点`})
            CALL apoc.create.relationship(n1, row.`合并后的边类型`, {}, n2) YIELD rel
            RETURN rel;
        """)

# 调用导入数据的函数
import_data(driver)

# 关闭驱动连接
driver.close()
