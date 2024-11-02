import os
import glob
import csv
from openai import OpenAI
from zhipuai import ZhipuAI

# 定义节点和边抽取函数
def get_gpt_response(prompt=""):
    API_SECRET_KEY = "sk-zk2b1f3a61aa82cee0eabc045237fd0bbe866cd9cc4db8f6"
    BASE_URL = "https://api.zhizengzeng.com/v1"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0
    )
    result = response.choices[0].message.content
    return result

def extract_nodes_and_edges_from_text(content):
    # 定义节点抽取的 prompt
    prompt_node = """
        ——————目标——————
        你是知识图谱的结点抽取助手，你需要从指定的输入文本中抽取出相应的结点信息以及结点所对应的类型。
        ——————规则——————
        你需要遵守以下的规则：
        （1）你需要根据输入的原始文本抽取出两方面的内容，第一是结点的名称，第二是结点所对应的类型。
        （2）对于结点的名称，你必须要做到与原始文本的内容一致，即你不能够凭空生成任何信息。
        （3）由于目前涉及到对医学疾病、检查指标、药物治疗方案等领域的知识图谱的抽取工作，因此在抽取关于结点的类型，你需要根据上述的领域以及输入的原始文本生成合适的结点类型，但是具体的结点类型不能够离开这几门大的领域。
        （4）你需要以一个一维数组的格式列出你抽取出来的所有结点，其中，数组里面的每一个元素为一个元组，元组里面包含结点的名称以及类型。格式示例如下：[(结点名称1,结点类型1),(结点名称2,结点类型2)]
        （5）除了规则（4）所指定的输出以外，你不能够输出其他任何无关信息，注意一维数组里面的每一个元素间不能有任何符号，例如换行符等，元组内部的内部的文字也不能够有任何符号所包围
        ——————输入文本——————
        输入文本：{}
    """.format(content)

    # 获取节点信息
    result_node = get_gpt_response(prompt=prompt_node)
    input_str = result_node.strip('[]')
    elements = input_str.split('),(')
    node_list = []
    for element in elements:
        element = element.strip('()')
        node, node_type = element.split(',', 1)
        node = node.strip()
        node_type = node_type.strip()
        node_list.append((node, node_type))

    # 定义边抽取的 prompt
    prompt_edge = """
        ——————目标——————
        你是一个知识图谱的边抽取助手，你需要从指定的输入文本以及给定的结点输入中抽取出结点与结点间的关联。
        ——————规则——————
        你需要遵守以下的规则：
        （1）你需要根据输入的文本以及结点抽取出由结点和边构成的三元组，代表了结点与边之间的关联。
        （2）你需要根据输入的结点以及原始文本自行定义边的类型，并从原始文本里面提取出结点与边之间的关联。其具体关联形式为如下的三元组形式：(源节点名称,边的类型,目标节点名称)。其中，源结点、目标结点名称务必与给定的结点名称一致，边的类型务必与你总结出来的边的类型一致。
        （3）再按照规则（2）抽取出指定的三元组后，你需要输出一个一维数组用来表示抽取出来的三元组，不能够输出其余不符合规定的信息。其具体格式如下：[(源节点名称1,边的类型1,目标节点名称1),(源节点名称2,边的类型2,目标节点名称2)]
        （4）在确定了一维数组以后，你的一维数组输出时每一个元素之间不能有其他符号，例如换行符等
        （5）你再遵守上述4条规则的基础上，需要尽可能地去使用给定的结点列表里的每一个结点，将其作为源节点以及目标节点，尽可能多地去挖掘结点之间的关系。注意，你需要尽可能多地根据给定的结点列表，使用不同的结点来作为三元组中的源节点，不要仅仅最后抽取出来的三元组里面只有一两个不同名称的源节点！
        ——————输入——————
        原始文本：{}
        输入的结点：{}
    """.format(content, result_node)

    # 获取边信息
    result_edge = get_gpt_response(prompt=prompt_edge)
    input_str_edge = result_edge.strip('[]')
    elements = input_str_edge.split('),(')
    edge_list = []
    for element in elements:
        element = element.strip('()')
        items = [item.strip() for item in element.split(',')]
        edge_list.append(tuple(items))

    return node_list, edge_list

# 读取目录下的所有文本文件
input_directory = '/home/zhuyifan/Cyan_A40/graph_extract/input/'
output_directory = '/home/zhuyifan/Cyan_A40/graph_extract/input/long_text_graph/'
txt_files = glob.glob(os.path.join(input_directory, '*.txt'))

# 合并所有文件的结果
all_nodes = []
all_edges = []

# 对每个文件进行节点和边的抽取
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as file:
        content = file.read()
    nodes, edges = extract_nodes_and_edges_from_text(content)
    all_nodes.extend(nodes)
    all_edges.extend(edges)

# 将所有节点和边写入 CSV 文件
with open(os.path.join(output_directory, 'nodes.csv'), 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['节点名称', '节点类型'])
    writer.writerows(all_nodes)

with open(os.path.join(output_directory, 'edges.csv'), 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['源节点', '边类型', '目标节点'])
    writer.writerows(all_edges)
