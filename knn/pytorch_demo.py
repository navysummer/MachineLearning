import numpy as np
from numpy import array, tile
import torch
from tqdm import tqdm


def classify0(inx, data_set, labels, k):
    dats_set_size = data_set.shape[0]
    test_x = torch.Tensor(tile(inx, (dats_set_size, 1)))
    tran_x = torch.Tensor(data_set)
    sort_class_count = []
    for x in tqdm(test_x):
        dists = []
        for y in tran_x:
            distance = torch.sum((x - y) ** 2) ** 0.5
            dists.append(distance.view(1))

        idxes = torch.cat(dists).argsort()[:k]
        unique, counts = np.unique(np.array([labels[idx] for idx in idxes]), return_counts=True)
        class_count_map = dict(zip(unique, counts))
        sort_class_count = sorted(class_count_map.items(), key=lambda d: d[1], reverse=True)
        # 返回预测结果
    if sort_class_count:
        return sort_class_count[0][0]
    else:
        return None


def create_data_set_label():
    data_set = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return data_set, labels


if __name__ == "__main__":
    data_set, labels = create_data_set_label()
    inx = [0, 0]
    predict = classify0(inx, data_set, labels, 3)
    print(predict)
