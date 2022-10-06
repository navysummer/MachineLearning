import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor



def create_data_set_label():
    data_set = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return data_set, labels


def retrieve_tree(i):
    list_of_tree = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                    {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return list_of_tree[i]

if __name__ == "__main__":
    # my_data_set, labels = create_data_set_label()
    # my_tree0 = retrieve_tree(0)
    #
    # reg = DecisionTreeRegressor(criterion='mse')
    # dt = reg.fit(my_tree0, labels)
    # x_test = [1, 0]
    # y_hat = dt.predict(x_test)
    # print(y_hat)
    n = 500
    x = np.random.rand(n) * 8 - 3
    x.sort()
    y = np.cos(x) + np.sin(x) + np.random.randn(n) * 0.4
    x = x.reshape(-1, 1)

    reg = DecisionTreeRegressor(criterion='mse')
    dt = reg.fit(x, y)
    x_test = np.linspace(-3, 5, 100).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.figure(facecolor="w")
    plt.plot(x, y, 'ro', label="actual")
    plt.plot(x_test, y_hat, 'k*', label="predict")
    plt.legend(loc="best")
    plt.title(u'Decision Tree', fontsize=17)
    plt.tight_layout()
    plt.grid()
    plt.show()
