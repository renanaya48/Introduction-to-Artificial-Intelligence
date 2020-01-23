import math
from collections import Counter
import sys
#from py_ex2 import get_data_cross, do_algo

K_FOLD = 5

class Tree(object):
    def __init__(self, att_name='root', poss_values=None, children_nodes=None):
        self.att_name = att_name
        self.poss_vals = dict()
        if children_nodes is not None and poss_values is not None:
            for val, child in zip(poss_values, children_nodes):
                self.add_child(val, child)

    def __repr__(self):
        return self.att_name

    def add_child(self, value, node):
        """
        Add a child node to the tree
        :param value: the title of the child: an optional value for this tree's root attribute
        :param node: the sub-tree rooted in the child
        :return: nothing
        """
        assert isinstance(node, Tree)
        self.poss_vals[value] = node

    def get_next_node(self, val):
        """
        return the sub tree rooted in the value, which is a possible assignment for this root's attribute
        :param val: the possible value
        :return: the sub-tree rooted in value if it exists, or None otherwise
        """
        if self.poss_vals.__contains__(val):
            return self.poss_vals[val]
        else:
            return None

    def is_leaf(self):
        """
        :return: true if the node is a leaf
        """
        return not self.poss_vals


def entropy(data):
    """
    calculate the entropy of a list of classes
    :param data: a list of values
    :return: the entropy of the items in the list
    """
    if len(data) <= 1:
        return 0
    counts = Counter(data)
    ent = 0
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= (p * math.log(p, 2))
    return ent

def make_decision_tree(data, attributes, att_classes_names, default):
    """
    Create a decision tree to fit the trainind data
    :param data: the training set, ordered by attributes
    :param attributes: the names of the different attributes (including the classification)
    :param att_classes_names: a list of lists of the possible values for each attribute (including the classification)
    :param default: The most common classification label
    :return:
    """
    # data is organized by the different attributes: a list of 'n_att' lists, where each list is 'n_examples' long
    if data is None or attributes is None or att_classes_names is None:
        return Tree(default)
    classifications = data[-1]
    if len(set(classifications)) == 1:
        return Tree(classifications[0])
    if len(attributes) == 1:
        return Tree(Counter(classifications).most_common(1)[0][0])
    # choose the attribute with the highest information gain
    len_data = len(classifications)
    min_ent = sys.maxsize
    best_att_index = -1

    for i in range(len(attributes) - 1):
        att_examples = data[i]
        total_vals_entropy = 0
        for poss_val in att_classes_names[i]:
            idx = [i for i, val in enumerate(att_examples) if val == poss_val]
            poss_val_classif = [classifications[j] for j in idx]
            poss_val_entropy = entropy(poss_val_classif)
            total_vals_entropy += poss_val_entropy * (len(idx) / len_data)
        if total_vals_entropy < min_ent:
            min_ent = total_vals_entropy
            best_att_index = i
    # the best attribute is attribute index of min entropy
    children_nodes = []
    new_atts = list(attributes)
    new_atts_poss_vals = list(att_classes_names)
    del new_atts[best_att_index]
    del new_atts_poss_vals[best_att_index]

    for poss_val in att_classes_names[best_att_index]:
        new_data_idx = [i for i, v in enumerate(data[best_att_index]) if v == poss_val]
        if len(new_data_idx) == 0:
            new_data = None
            default_class = 'yes'
            children_nodes.append(make_decision_tree(new_data, new_atts, new_atts_poss_vals, default_class))
            continue
        new_data = list()
        for i in range(len(attributes)):
            if i == best_att_index:
                continue
            new_data.append([data[i][j] for j in new_data_idx])
        poss_classes = Counter(new_data[-1])
        default_class = poss_classes.most_common(1)[0][0]
        children_nodes.append(make_decision_tree(new_data, new_atts, new_atts_poss_vals, default_class))
    return Tree(attributes[best_att_index], att_classes_names[best_att_index], children_nodes)

def print_tree(dt):
    """
    Print the decision tree to the output_tree.txt file
    :param dt: a decision tree
    :return: the output file
    """
    #out_file = open("tree.txt", 'w')
    out_file = open("output.txt", 'w')
    print_sub_tree(out_file, dt)
    out_file.close()


def print_sub_tree(file_object, tree, n_tabs=0, line=False):
    """
    A recursive function to print a sub tree
    :param file_object: the file to print the output to
    :param tree: the root of the sub-tree to print
    :param n_tabs: number of tabs to start the current line printing with. (increases with each generation)
    :param line: True if there needs to be a line printed in the beginning. for the nodes that are not in depth 1 this will be True.
    :return:
    """
    if tree is None:
        return
    for val in sorted(tree.poss_vals):
        for _ in range(n_tabs):
            file_object.write('\t')
        if line:
            file_object.write('|')
        if tree.poss_vals[val].is_leaf():
            file_object.write(tree.att_name + '=' + val + ':' + tree.poss_vals[val].att_name+'\n')
        else:
            file_object.write(tree.att_name+'='+val+'\n')
            print_sub_tree(file_object, tree.poss_vals[val], n_tabs+1, True)

def test_decision_tree(tree, test, attributes, att_poss_vals):
    """
    test.txt the accuracy of the decision tree on the test.txt set
    :param tree: the decision tree that was based on the train.txt data
    :param test: the test.txt set, ordered by examples
    :param attributes: the list of attributes names
    :param att_poss_vals: list of lists of possible attributes values
    :return: the prediction and accuracy
    """
    # the test.txt set is given as a list of all test.txt examples, where each list is made up of a list of features
    correct = 0
    pred = []
    default = Counter(att_poss_vals[-1]).most_common(1)[0][0]
    for test_example in test:
        curr_node = tree
        while curr_node is not None and not curr_node.is_leaf():
            curr_att_name = curr_node.att_name
            att_index = attributes.index(curr_att_name)
            curr_feat = test_example[att_index]
            curr_node = curr_node.get_next_node(curr_feat)
        if curr_node is None:
            curr_pred = default
        else:
            curr_pred = curr_node.att_name
        pred.append(curr_pred)
        if curr_pred == test_example[-1]:
            correct += 1
    acc = '{:>.02f}'.format(correct / len(test))
    return acc

def get_tree_data(train_filename='dataset.txt'):
    """
    extract the train.txt information from the dataset
    :param train_filename: the name of the training data file
    :return: the train.txt data organized in two ways: listed by example number, or listed by attributes.
    (the train.txt are returned in both dimensions: n_attribute X n_examples and n_examples X n_attribute)
    """
    train_file = open(train_filename)
    all_train_lines = train_file.readlines()
    attributes = [w.strip() for w in all_train_lines[0].split('\t')]
    n_att = len(attributes)
    data_examples = [[w.strip() for w in line.split('\t')] for line in all_train_lines[1:]]
    data_attributes = [[example[i] for example in data_examples] for i in range(n_att)]
    att_classes_names = [set(classes) for classes in data_attributes]
    return data_examples, data_attributes, attributes, att_classes_names


def main():
    acc_tree = []
    train, data_attributes, attributes, att_classes_names = get_tree_data()
    dt = make_decision_tree(data_attributes, attributes, att_classes_names,
                            Counter(data_attributes[-1]).most_common(1)[0][0])
    print_tree(dt)
    """
    for i in range(K_FOLD):
        train.txt, data_attributes, test_examples, attributes, att_classes_names = get_data_cross()

        dt = make_decision_tree(data_attributes, attributes, att_classes_names,
                                Counter(data_attributes[-1]).most_common(1)[0][0])
        acc_tree.append(float(test_decision_tree(dt, test_examples, attributes, att_classes_names)))
    avg_acc_tree = '{:>.02f}'.format(sum(acc_tree) / K_FOLD)
    print(avg_acc_tree)
    """



if __name__ == "__main__":
    main()