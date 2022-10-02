from numpy import tile, array


def classify0(inx: array, data_set: array, labels: array, k: int):
    """
    原理：使用欧式距离公式：((xa0-xb0)**2-(xa1-xb1)**2)**0.5计算出输入点到各个点的欧式距离，按照距离从小到大排列，取出前前k个的类别出现的
         频率，频率最高的一个的类别即为预测分类
    :param data_set: 训练的数据集
    :param inx: 输入向量
    :param labels: 标签向量
    :param k: 最近邻数量
    :return:
    """
    # 通过shape获取行和列，返回的是包含两个元素的元组，即(行数，列数)
    dats_set_size = data_set.shape[0]
    # numpy.tile(data,(x,y)) 将数据data扩展到x行y列
    # 计算输入向量和训练集的距离差，即x1-x0
    diff_mat = tile(inx, (dats_set_size, 1)) - data_set
    # 坐标差的平方，即(x1-x0)^2
    square_diff_mat = diff_mat ** 2
    # 对坐标差的平方进行求和
    square_diff_mat_sum = square_diff_mat.sum(axis=1)
    # 欧式距离
    distance = square_diff_mat_sum ** 0.5
    # 对距离进行排序
    sort_distance = distance.argsort()
    # 训练数据
    class_count_map = dict()
    for i in range(k):
        vote_label = labels[sort_distance[i]]
        class_count_map.update({vote_label: class_count_map.get(vote_label, 0) + 1})
    # 训练的结果
    sort_class_count = sorted(class_count_map.items(), key=lambda x: x[1], reverse=True)
    # 返回预测结果
    return sort_class_count[0][0]


def create_data_set_label():
    data_set = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return data_set, labels


if __name__ == "__main__":
    data_set, labels = create_data_set_label()
    inx = [0, 0]
    predict = classify0(inx, data_set, labels, 3)
    print(predict)
