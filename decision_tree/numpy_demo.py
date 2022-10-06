from math import log2
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

matplotlib.rc("font", family='WenQuanYi Zen Hei')


# font_set = FontProperties(fname=r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", size=15)


def calc_shannon_ent(data_set):
    """
    数据的香农熵计算公式：sum(-p*log2(p))
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_count_map = dict()
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_count_map:
            label_count_map.update({current_label: 0})
        label_count_map[current_label] += 1
    shannon_ent = 0.0
    for key, value in label_count_map.items():
        prob = float(value) / num_entries
        shannon_ent -= prob * log2(prob)
    return shannon_ent


def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count_map = dict()
    for vote in class_list:
        if vote not in class_count_map:
            class_count_map.update({vote: 0})
        class_count_map[vote] += 1
    sorted_class_count = sorted(class_count_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


def creat_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = creat_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def create_plot(tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    subplot = plt.subplot(111, frameon=False, **ax_props)
    total_width = float(get_num_leafs(tree))
    total_depth = float(get_tree_depth(tree))
    x_offset = -0.5 / total_width
    y_offset = 1.0
    plot_tree(subplot, tree, (0.5, 1.0), "", x_offset, y_offset, total_width, total_depth)
    plt.show()


def plot_node(subplot, node_text, center_point, parent_point, node_type):
    subplot.annotate(node_text, xy=parent_point, xycoords="axes fraction", xytext=center_point,
                     textcoords="axes fraction", va="center", ha="center", bbox=node_type,
                     arrowprops=dict(arrowstyle="<-"))


def plot_mid_text(subplot, center_point, parent_point, text):
    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    subplot.text(x_mid, y_mid, text)


def plot_tree(subplot, tree, parent_point, node_text, x_offset, y_offset, total_width, total_depth):
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_str = list(tree.keys())[0]
    center_point = (x_offset + (1.0 + float(num_leafs))) / 2.0 / total_width, y_offset
    plot_mid_text(subplot, center_point, parent_point, node_text)
    plot_node(subplot, first_str, center_point, parent_point, dict(boxstyle="sawtooth", fc="0.8"))
    second_dict = tree[first_str]
    y_offset = y_offset - 1.0 / total_depth
    for key, value in second_dict.items():
        if isinstance(value, dict):
            x_offset, y_offset = plot_tree(subplot, value, center_point, str(key), x_offset, y_offset, total_width,
                                           total_depth)
        else:
            x_offset = x_offset + 1.0 / total_width
            plot_node(subplot, value, (x_offset, y_offset), center_point, dict(boxstyle="round4", fc="0.8"))
            plot_mid_text(subplot, (x_offset, y_offset), center_point, str(key))
    y_offset = y_offset + 1.0 / total_width
    return x_offset, y_offset


def get_num_leafs(tree):
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key, value in second_dict.items():
        if isinstance(value, dict):
            num_leafs += get_num_leafs(value)
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(tree):
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key, value in second_dict.items():
        if isinstance(value, dict):
            max_depth = 1 + get_tree_depth(value) if 1 + get_tree_depth(value) > max_depth else max_depth
        else:
            max_depth = max_depth if max_depth > 1 else 1
    return max_depth


def create_data_set_label():
    data_set = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return data_set, labels


def retrieve_tree(i):
    list_of_tree = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                    {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return list_of_tree[i]


def classify(input_tree, feat_labels, test_vec):
    first = list(input_tree.keys())[0]
    second_dict = input_tree[first]
    feat_index = feat_labels.index(first)
    class_label = None
    for key, value in second_dict.items():
        if test_vec[feat_index] == key:
            if isinstance(value, dict):
                class_label = classify(value, feat_labels, test_vec)
            elif isinstance(value, list):
                for x in value:
                    class_label = classify(x, feat_labels, test_vec)
            else:
                class_label = value
    return class_label


if __name__ == "__main__":
    my_data_set, labels = create_data_set_label()
    my_tree0 = retrieve_tree(0)
    create_plot(my_tree0)
    print(classify(my_tree0, labels, [1, 0]))
    print(classify(my_tree0, labels, [1, 1]))
