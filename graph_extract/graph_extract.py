from openai import OpenAI
import csv
def get_gpt_response(prompt=""):
    API_SECRET_KEY = "sk-zk2844244f76f82c1bcbe90a4c8c6fb895a66cc4f08df87d"
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

with open('/home/zhuyifan/Cyan_A40/graph_extract/input/test.txt', 'r', encoding='utf-8') as file:
    content = file.read()
prompt = """
        ——————目标——————
        你是知识图谱的结点抽取助手，你需要从指定的输入文本中抽取出相应的结点信息以及结点所对应的类型。
        ——————规则——————
        你需要遵守以下的规则：
        （1）你需要根据输入的原始文本抽取出两方面的内容，第一是结点的名称，第二是结点所对应的类型。
        （2）对于结点的名称，你必须要做到与原始文本的内容一致，即你不能够凭空生成任何信息。
        （3）对于结点的类型，你需要从以下的几个类型中选择一个作为当前结点的合适类型，你绝不能够凭空编造。目前的类型如下：疾病、症状、疾病定义、诊断指标、治疗方案
        （4）你需要以一个一维数组的格式列出你抽取出来的所有结点，其中，数组里面的每一个元素为一个元组，元组里面包含结点的名称以及类型。格式示例如下：[(结点名称1,结点类型1),(结点名称2,结点类型2)]
        （5）除了规则（4）所指定的输出以外，你不能够输出其他任何无关信息，注意一维数组里面的每一个元素间不能有任何符号，例如换行符等，元组内部的内部的文字也不能够有任何符号所包围
        ——————输入文本——————
        输入文本：{}
    """
result_node = get_gpt_response(prompt=prompt.format(content))
# print(result_node)
# 去掉最外层的方括号
input_str = result_node.strip('[]')

# 使用正则表达式或简单分隔方法分割各个元素
elements = input_str.split('),(')

# 结果列表
node_list = []

# 遍历分割的元素
for element in elements:
    # 去除可能残留的括号
    element = element.strip('()')
    
    # 分割结点名称和类型
    node, node_type = element.split(',', 1)
    
    # 去除结点名称和类型中的空格
    node = node.strip()
    node_type = node_type.strip()
    
    # 将结点名称和类型组成元组，并加入结果列表
    node_list.append((node, node_type))

# 输出解析结果
print(node_list)
prompt_edge = """"
        ——————目标——————
        你是一个知识图谱的边抽取助手，你需要从指定的输入文本以及给定的结点输入中抽取出结点与结点间的关联。
        ——————规则——————
        你需要遵守以下的规则：
        （1）你需要根据输入的文本以及结点抽取出由结点和边构成的三元组，代表了结点与边之间的关联。
        ——————边的定义——————
        (疾病 表现为 症状)
        (疾病 定义为 疾病定义)
        (疾病 诊断通过 诊断指标)
        (疾病 治疗 治疗方案)
        (症状 评估通过 诊断指标)
        (症状 缓解为 治疗方案)
        (诊断指标 指标定义 疾病定义)
        (治疗方案 应用于 疾病定义)
        (疾病 包含 疾病)
        ————————————————
        （2）你需要根据输入的结点以及上述边的定义里面提取出结点与边之间的关联。其具体关联形式为如下的三元组形式：(源节点名称,边的类型,目标节点名称)。其中，源结点、目标结点名称务必与输入一致，边的类型务必与上述边的定义里面每个三元组里面的第二个元素一致。
        （3）再按照规则（2）抽取出指定的三元组后，你需要输出一个一维数组用来表示抽取出来的三元组，不能够输出其余不符合规定的信息。其具体格式如下：[(源节点名称1,边的类型1,目标节点名称1),(源节点名称2,边的类型2,目标节点名称2)]
        （4）在确定了一维数组以后，你的一维数组输出时每一个元素之间不能有其他符号，例如换行符等
        ——————输入——————
        原始文本：{}
        输入的结点：{}
    """
result_edge = get_gpt_response(prompt=prompt_edge.format(content,result_node))
# print(result_edge)
input_str_edge = result_edge.strip('[]')

# 使用正则表达式或简单分隔方法分割各个元素
elements = input_str_edge.split('),(')

# 结果列表
edge_list = []

# 遍历分割的元素
for element in elements:
    # 去除可能残留的括号
    element = element.strip('()')
    
    # 分割三元组元素
    items = [item.strip() for item in element.split(',')]
    
    # 将三元组元素组成元组，并加入结果列表
    edge_list.append(tuple(items))

# 输出解析结果
print(edge_list)
with open('/home/zhuyifan/Cyan_A40/graph_extract/input/nodes.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['节点名称', '节点类型'])
    writer.writerows(node_list)

# 将第二个数组写入 CSV 文件
with open('/home/zhuyifan/Cyan_A40/graph_extract/input/edges.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['源节点', '边类型', '目标节点'])
    writer.writerows(edge_list)