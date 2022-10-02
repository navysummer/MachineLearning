import numpy
import tensorflow as tf

from numpy import array, tile


def get_data_distance(data, input_data):
    dats_size = data.shape[0]
    diff_mat = tf.subtract(data, tile(input_data, (dats_size, 1)))
    square_diff_mat = tf.square(diff_mat)
    square_diff_mat_sum = tf.reduce_sum(square_diff_mat, axis=1)
    distance = tf.sqrt(square_diff_mat_sum)
    return data, distance


def classify0(data_set: array, labels: array, inx, k: int):
    train_dataset = tf.data.Dataset.from_tensor_slices(data_set)
    train_dataset = train_dataset.map(map_func=lambda d: get_data_distance(d, inx))
    distance = []
    for x in train_dataset.as_numpy_iterator():
        distance.append(x[1][0])
    distance = numpy.array(distance)
    sort_distance = distance.argsort()

    class_count_map = dict()
    for i in range(k):
        vote_label = labels[sort_distance[i]]
        class_count_map.update({vote_label: class_count_map.get(vote_label, 0) + 1})
    sort_class_count = sorted(class_count_map.items(), key=lambda m: m[1], reverse=True)
    return sort_class_count[0][0]


def create_data_set_label():
    data_array = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label_array = ["A", "A", "B", "B"]
    return data_array, label_array


if __name__ == "__main__":
    data_arr, label_arr = create_data_set_label()
    x = [0, 0]
    predict = classify0(data_arr, label_arr, x, 3)
    print(predict)
