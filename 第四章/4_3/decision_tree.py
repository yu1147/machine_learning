# 决策树生成

# 定义结点类


class Node(object):
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init

# 建立决策树


def Treegen(df):
    # 先生成根节点
    new_node = Node(None, None, {})
    label_arr = df[df.columns[-1]]       # 得出标记（如好瓜坏瓜）
    label_count = labelNode(label_arr)   # labelNode（）函数得到记录好瓜坏瓜数量的字典

    if label_count:                                              # 标记存在的情况下
        new_node.label = max(label_count, key=label_count.get)  # 比较得出最大值
        if len(label_count) == 1 or len(label_arr) == 0:  # 如果只有一种属性是直接返回node
            return new_node

        #  决策树第三种情况，应该先选出最优attr（属性）
        new_node.attr, div_value = optAttr(df)  # optAttr()求出最优属性

        # 递推建树
        # 离散值
        if div_value == 0:
            value_count = valueCount(df[new_node.attr])
            for value in value_count:
                df_v = df[df[new_node.attr].isin([value])]
                df_v = df_v.drop(new_node.attr, 1)
                new_node.attr_down[value] = Treegen(df_v)

        else:
            value_l = "<=%.3f"%div_value
            value_r = ">#.3f"%div_value
            df_v_l = df[df[new_node.attr] <= div_value]
            df_v_r = df[df[new_node.attr] > div_value]
            new_node.attr_down[value_l] = Treegen(df_v_l)
            new_node.attr_down[value_r] = Treegen(df_v_r)
    return new_node




'''
最优属性，ID3求出最大信息增益
'''
def optAttr(df):
    info_gain = 0

    for attr_id in df.columns[1:-1]:
        info_gain_temp, div_value_temp = infoGain(df, attr_id)  # infoGain()求出属性的熵值
        if info_gain_temp > info_gain:
            info_gain = info_gain_temp
            div_value = div_value_temp
            opt_attr = attr_id

    return opt_attr, div_value


'''
求出属性的信息增益
'''
def infoGain(df, attr_id):
    info_gain = infoEnt(df.values[:, -1])  # df.columns[-1]
    div_value = 0  # div_value=0说明为离散值，=t时说明为连续值
    n = len(df[attr_id])  # 样本数量
    # 1、连续值
    if df[attr_id].dtype == (str, float):
        sub_info_ent = {}  # 存储连续值的熵
        # 排序，排序后编号会乱序，因此可以去前面的编号，重新编制，即更新编号
        df = df.sort([attr_id], ascending=1)
        df = df.reset_index(drop=True)

        # 求熵
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]

        for i in range(n-1):
            div_value_temp = (data_arr[i]+data_arr[i+1])/2
            sub_info_ent[div_value_temp] = ((i+1)*infoEnt(label_arr[0:i+1])/n+(n-i-1))*(infoEnt(label_arr[i+1:-1])/n)
        div_value, sub_info_ent_max = min(sub_info_ent.items(), key=lambda x: x[1])
        info_gain -= sub_info_ent_max

    # 2、离散值
    else:
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]
        value_count = valueCount(data_arr)

        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key]*infoEnt(key_label_arr)/n
    return info_gain, div_value


'''
求出Ent（D)
'''
def infoEnt(label_arr):
    from math import  log2
    ent = 0
    label_count = labelNode(label_arr)
    n = len(label_arr)
    for label in label_count:
        ent -= (label_count[label]/n)*log2(label_count[label]/n)

    return ent


'''
labelNode（）函数得到记录好瓜坏瓜数量的字典
'''
def labelNode(label_arr):
    label_count = {}
    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count


'''
得到属性字典
'''
def valueCount(data_arr):
    value_count = {}
    for label in data_arr:
        if label in value_count:
            value_count[label] += 1
        else:
            value_count[label] = 1
    return value_count


'''
预测
'''
def predict(root, df_sample):
    import re
    while root.attr != None:
        # 连续值
        if df_sample[root.attr] == (float, int):
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                break
            if df_sample[root.attr].values[0] <= div_value:
                key = "<=%.3f"%div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f"%div_value
                root = root.attr_down[key]
        # 离散值
        else:
            key = df_sample[root.attr].values[0]
                # check whether the attr_value in the child branch
            if key in root.attr_down:
                root = root.attr_down[key]
            else:
                break
    return root.label


'''
画树
'''
def drawPng(root, out_file):
    from pydotplus import graphviz
    g = graphviz.Dot()
    Tree2Graph(0, g, root)
    g2 = graphviz.graph_from_dot_data(g.to_string())
    g2.write_png(out_file)



def Tree2Graph(i, g, root):
    from pydotplus import graphviz
    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label))

    for value in list(root.attr_down):
        i, g_child = Tree2Graph(i + 1, g, root.attr_down[value])
        g.add_edge(graphviz.Edge(g_node, g_child, label=value))

    return i, g_node
