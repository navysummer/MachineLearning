from numpy import array, tile
from sklearn.neighbors import KNeighborsClassifier


def classify0(inx, data_set, labels, k):
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(data_set, labels)
    x_predict = array(inx).reshape(1, -1)
    y_predict = kNN_classifier.predict(x_predict)
    return y_predict[0]


def create_data_set_label():
    data_set = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return data_set, labels


if __name__ == "__main__":
    data_set, labels = create_data_set_label()
    inx = [0, 0]
    predict = classify0(inx, data_set, labels, 3)
    print(predict)
