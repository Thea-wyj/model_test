import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
import collections
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pydot
import pandas as pd

# 画图每个类别节点的填充色
color_list = ['#3ea0e6', '#e99355']
"""
树结构：
节点标号从0开始 先序遍历
node_count：int
threshold：numpy.ndarray 
feature.txt：numpy.ndarray
children_left：numpy.ndarray
children_right：numpy.ndarray
label:节点对应的类别 numpy.ndarray
注意：feature.txt and threshold only apply to split nodes.
"""


class Tree_:
    def __init__(self, node_count=0, threshold=None, feature=None, children_left=None, children_right=None, label=None):
        self.node_count = node_count
        self.threshold = threshold
        self.feature = feature
        self.children_left = children_left
        self.children_right = children_right
        self.label = label


class TreeNode:
    def __init__(self, num=0, feature=0, threshold=0, left=None, right=None, up=None, label=None, angle=None,
                 fill_color=None):
        self.num = num
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.up = up
        self.label = label
        self.angle = angle
        self.fill_color = fill_color


# 构建树 返回根节点
def buildTree(tree_):
    treenode_list = []
    for i in range(tree_.node_count):
        # treenode = TreeNode(i, tree_.feature[i], round(tree_.threshold[i], 3), tree_.children_left[i],
        #                     tree_.children_right[i], None, tree_.label[i])
        treenode = TreeNode(i, tree_.feature[i], (tree_.threshold[i]), tree_.children_left[i],
                            tree_.children_right[i], None, tree_.label[i])
        treenode_list.append(treenode)
    for i in range(tree_.node_count):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
            treenode.left.angle = 45
            treenode_list[treenode.left.num].up = treenode

        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]
            treenode.right.angle = -45
            treenode_list[treenode.right.num].up = treenode
    return treenode_list[0]


'''
读决策树信息文件
返回值包含：
node_count：int
threshold：numpy.ndarray 
feature.txt：numpy.ndarray
children_left：numpy.ndarray
children_right：numpy.ndarray
label:节点对应的类别 numpy.ndarray
'''


def read_dataset(filename):
    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    dataset = []
    for line in all_lines[0:6]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        line_ = np.array(list(map(float, line)))
        dataset.append(line_)
    dataset[0] = int(dataset[0])  # node_count转为int 其余为numpy.array
    # 除了threshold的numpy里面存储的类型为float 其余的numpy里面存储的类型都为int
    dataset[2] = dataset[2].astype(int)
    dataset[3] = dataset[3].astype(int)
    dataset[4] = dataset[4].astype(int)
    dataset[5] = dataset[5].astype(int)
    return dataset


# 输入存储树信息的文件路径 返回封装好的tree_结构
def getTree(filepath):
    result = read_dataset(filepath)
    tree_ = Tree_(result[0], result[1], result[2], result[3], result[4], result[5])
    return tree_


# 读决策树信息文件 获得特征文本
def getFeatureText(filepath):
    fr = open(filepath, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    feature_text = []
    for line in all_lines[6:7]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        for word in line:
            feature_text.append(word)
    return feature_text


# 读决策树信息文件 获得分类的类别标签文本
def getLabelText(filepath):
    fr = open(filepath, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    label_text = []
    for line in all_lines[7:8]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        for word in line:
            label_text.append(word)
    return label_text


# 计算APL (训练集上样本的平均决策路径)
def average_path_length(tree, X, class_dict):
    """Compute average path length: cost of simulating the average
    example; this is used in the objective function.

    @param tree: DecisionTreeClassifier instance
    @param X: NumPy array (D x N)
              D := number of dimensions
              N := number of examples
    @return path_length: float
                         average path length
    """
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    # leaf_indices = tree.apply(X)
    # leaf_indices = tree.apply(X)#获得样本对应的叶子节点编号序列 https://www.zhihu.com/question/39254529
    # leaf_counts = np.bincount(leaf_indices)#统计上面每个叶子节点编号出现的次数
    node_depth = np.zeros(shape=class_dict.shape, dtype=np.int64)
    is_leaves = np.zeros(shape=class_dict.shape, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if ((children_left[node_id] != children_right[node_id]) and children_left[node_id] != -1):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    path_length = np.dot(node_depth, class_dict) / float(X.shape[0])
    return path_length


def countNodeCount(tree):
    # 与children_left[i], children_right[i]相关的建议仍旧使用自带的node_count属性
    count = 1
    for i in tree.children_right:
        if i != -1:
            count += 1
    for i in tree.children_left:
        if i != -1:
            count += 1
    return count


def collect(node, counter):
    if node == -1 or (node.left == -1 and node.right == -1):
        return "#"
    serial = "{},{},{}".format(node.feature, collect(node.left, counter),
                               collect(node.right, counter))
    counter[serial] += 1
    return serial


def countDuplicateSubstree(tree):
    # 计算重复子树
    # print("start countDuplicateSubstree")
    counter = collections.Counter()  # 计数器
    duplicate_subtree = 0

    n_nodes = int(tree.node_count)
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    label = tree.label

    # 利用children_left和children_right构造TreeNode形式的树
    treenode_list = []
    for i in range(n_nodes):
        treenode = TreeNode(i, feature[i], round(threshold[i], 3), children_left[i], children_right[i], None, label[i])
        treenode_list.append(treenode)
    for i in range(n_nodes):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]

    # 将所有子树以[属性，阈值，左子树，右子树]进行序列化，在counter中统计每种序列化形式出现的次数
    collect(treenode_list[0], counter)

    # print(counter)

    # 出现次数大于等于2即表示有重复子树
    for item in counter:
        if counter[item] >= 2:
            duplicate_subtree += len(item.split(","))
            # duplicate_subtree += len(item.split(","))*counter[item]
    # print(duplicate_subtree*2)
    # print(countNodeCount(tree))
    # print(duplicate_subtree*2/countNodeCount(tree))
    # print("end countDuplicateSubstree")

    return duplicate_subtree * 2 / countNodeCount(tree)


def countAverageDuplicateAttr(tree, X, class_dict):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    label = tree.label
    # leaf_indices = tree.apply(X)#获得样本对应的叶子节点编号序列 https://www.zhihu.com/question/39254529
    # print("start countAverageDuplicateAttr")
    # print(leaf_indices)
    # leaf_counts = np.bincount(leaf_indices)#统计上面每个叶子节点编号出现的次数

    # print(leaf_counts)
    # print(children_left)
    # print(children_right)
    # print(feature.txt)
    threshold = tree.threshold

    # 利用children_left和children_right构造TreeNode形式的树
    treenode_list = []
    for i in range(n_nodes):
        treenode = TreeNode(i, feature[i], round(threshold[i], 3), children_left[i], children_right[i], None, label[i])
        treenode_list.append(treenode)
    for i in range(n_nodes):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
            treenode_list[treenode.left.num].up = treenode
        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]
            treenode_list[treenode.right.num].up = treenode

    duplicate_attr_list = []
    for i in range(class_dict.size):
        feature_dict = {}
        duplicate_attr = 0
        node_in_path = 0
        if class_dict[i] != 0:
            # print("i=",i,",value=",leaf_counts[i])
            t = treenode_list[i]
            # print("treenode:",t.num)
            while (t.up != None):
                feature_dict[t.up.feature] = feature_dict.setdefault(t.up.feature, 0) + 1
                t = t.up
                node_in_path += 1
            # print("feature_dict:",feature_dict)
            for key, value in feature_dict.items():
                if value >= 2 and key != -2:
                    duplicate_attr += value
            # print("node_in_path", node_in_path)
            # print("duplicate_attr:",duplicate_attr)
            # print("***")

        if node_in_path != 0:
            duplicate_attr_list.append(duplicate_attr / node_in_path)
        else:
            duplicate_attr_list.append(0.0)
    # print("duplicate_attr_list",duplicate_attr_list)
    # print("leaf_counts",leaf_counts)

    path_length = np.dot(duplicate_attr_list, class_dict) / float(X.shape[0])
    # print(path_length)
    # print("end countAverageDuplicateAttr")

    return path_length


'''
调用时
class_dict!=None class_dict获取落在每个节点上的样本数量
test_labels！=None test_labels存储数据集sample中样本对应的标签
'''


def getSampleClass(treenode, sample, class_dict, test_labels):
    if (treenode == -1):
        return
    featurn_index = treenode.feature
    if (sample[featurn_index] <= treenode.threshold):
        getSampleClass(treenode.left, sample, class_dict, test_labels)
    else:
        getSampleClass(treenode.right, sample, class_dict, test_labels)
    if (treenode.left == -1 and treenode.right == -1):
        if (class_dict != None):
            class_dict[treenode.num] = class_dict[treenode.num] + 1
        if (test_labels != None):
            test_labels.append(treenode.label)


def getSamplesClass(treenode, samples, class_dict, test_labels):
    for sample in samples:
        getSampleClass(treenode, sample, class_dict, test_labels)
    return class_dict


# 提取决策树的指标 写入index_path
def getDecisionTreeIndexes(tree_, X_train, Y_test_label, tree_test_label, class_dict, index_path):
    result = {}
    # 指标提取 除了稳定性指标是十折交叉验证取平均值外 其余指标都通过上面生成的决策树进行计算
    AUC = roc_auc_score(Y_test_label, tree_test_label)
    result['AUC'] = round(AUC, 3)

    APL = average_path_length(tree_, X_train, class_dict)
    result['APL'] = round(APL, 3)

    node_count = countNodeCount(tree_)
    result['node count'] = node_count

    duplicate_subtree = countDuplicateSubstree(tree_)
    result['duplicate_subtree'] = duplicate_subtree

    duplicate_attr = countAverageDuplicateAttr(tree_, X_train, class_dict)
    result['duplicate_attr'] = duplicate_attr

    return result


# 递归存储每个节点用于画图的文本
def visualizeCore(treenode, list, feature_text, label_text):
    if (treenode == -1):
        return
    # 叶子节点不显示特征和阈值
    if (treenode.left == -1 and treenode.right == -1):
        list.append(
            str(treenode.num) + ' [label="\\nclass=' + label_text[treenode.label] + '", fillcolor=' + '"' + color_list[
                treenode.label] + '"];')
    # 分支节点显示特征和阈值
    else:
        list.append(str(treenode.num) + ' [label="' + (feature_text[treenode.feature]) + '<=' + str(
            treenode.threshold) + '", fillcolor=' + '"' + color_list[treenode.label] + '"];')
    # 有父节点的 创建与父节点的连线
    if (treenode.up != None):
        if (treenode.up.num == 0 and treenode.up.left.num == treenode.num):
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + ', headlabel="True"];')
        elif (treenode.up.num == 0 and treenode.up.right.num == treenode.num):
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + ', headlabel="False"];')
        else:
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + '];')
    list.append('\n')
    visualizeCore(treenode.left, list, feature_text, label_text)
    visualizeCore(treenode.right, list, feature_text, label_text)


# 生成决策树信息文件对应决策树的画图文本
def visualize(treenode, feature_text, label_text, tree_dot_path):
    list = []
    list.append(
        'digraph Tree {node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;edge [fontname=helvetica] ;' + '\n')
    visualizeCore(treenode, list, feature_text, label_text)
    list.append('\n' + '}')
    with open(tree_dot_path, "w+") as f:
        f.write(''.join(list))


# 生成决策树png图片
# doc_path：用于画决策树的dot文件路径
# pic_path：生成决策树png图像的路径
def showTree(doc_path, pic_path):
    (graph,) = pydot.graph_from_dot_file(doc_path)
    graph.write_png(pic_path);


def tree_extract(path_dict):
    # 加载数据集
    """从dataset_test加载样本、特征列表"""
    dataset_test_path = path_dict["dataset_test"]
    dataset_train_path = path_dict["dataset_train"]
    csv_data = pd.read_csv(dataset_train_path, low_memory=False)  # 防止弹出警告
    X_train = np.array(csv_data)[:, :-1]
    csv_data = pd.read_csv(dataset_test_path, low_memory=False)  # 防止弹出警告
    X_test = np.array(csv_data)[:, :-1]
    """遍历全部XAI提取指标"""
    index_dict = {}
    for i in range(len(path_dict) - 2):
        xai = "XAI" + str(i + 1)
        blackbox_path = path_dict[xai]["blackbox"]
        proxy_path = path_dict[xai]["proxy"]
        """ 加载黑盒、决策树 """
        # 从blackbox_path加载黑盒，获取黑盒预测结果y_bb
        blackbox = joblib.load(blackbox_path)
        y_test = blackbox.predict(X_test)
        # 读取决策树文件
        decision_info_path = proxy_path  # 决策树信息文件存储地址
        tree_ = getTree(decision_info_path)
        feature_text = getFeatureText(decision_info_path)  # 获取特征文本（画图）
        lable_text = getLabelText(decision_info_path)  # 获取标签文本（画图）
        # 构建树结构 返回根节点
        treeNode = buildTree(tree_)
        # # 输出决策树画图用的dot文件
        # tree_dot_path = 't.dot'  # 决策树用于画图的dot文件
        # visualize(treeNode, feature_text, lable_text, tree_dot_path)
        # getSamplesClass 存储落在叶子节点上的样本数量 类似之前使用tree 下面两句的功能
        # leaf_indices = tree.apply(X)  # 获得样本对应的叶子节点编号序列 https://www.zhihu.com/question/39254529
        # leaf_counts = np.bincount(leaf_indices)  # 统计上面每个叶子节点编号出现的次数
        class_dict = [0] * tree_.node_count
        class_dict = getSamplesClass(treeNode, X_train, class_dict, None)
        # 存储测试集决策树预测的结果
        test_labels = []
        getSamplesClass(treeNode, X_test, None, test_labels)
        test_labels = np.array(list(map(int, test_labels)))
        class_dict = np.array(list(map(int, class_dict)))
        # 提取指标
        index_dict[xai] = getDecisionTreeIndexes(tree_, X_train, y_test, test_labels, class_dict, 1)
    # print(index_dict)
    return index_dict
