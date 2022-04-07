import numpy as np
import time
from sklearn.datasets import load_svmlight_file

allow_duplicate = False


class KDNode():
    def __init__(self, value, label, left, right, depth):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.depth = depth


class KDTree():
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels
        if(len(self.values) != len(self.labels)):
            raise ValueError("values and labels must be the same length")
        if(len(self.values) == 0):
            raise ValueError("values must be non-empty")
        self.dimension_len = len(self.values[0])
        data = np.column_stack((self.values, self.labels))
        self.root = self.build_KDTree(data, 0)
        self.knn_result = []
        self.calculation_nums = 0

    def distance(self, point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))

    def build_KDTree(self, data, depth):
        if(len(data) == 0):
            return None
        current_dimension = depth % self.dimension_len
        data = data[data[:, current_dimension].argsort()]
        median_index = len(data) // 2
        node = KDNode(data[median_index][:-1],
                      data[median_index][-1], None, None, depth)
        node.left = self.build_KDTree(data[:median_index], depth+1)
        node.right = self.build_KDTree(data[median_index+1:], depth+1)
        return node

    def search_knn(self, target, K):
        if(self.root is None):
            raise ValueError("tree is empty")
        if(K > len(self.values)):
            raise ValueError(
                "K must be less than or equal to the number of data points")
        if(len(target) != self.dimension_len):
            print('target dim:', len(target))
            print('tree dim:', self.dimension_len)
            raise ValueError(
                "target must be the same length as the dimension of the tree")
        self.knn_result = []
        self.calculation_nums = 0
        self.search_knn_recursive(self.root, target, K)
        return self.calculation_nums

    def search_knn_recursive(self, root, target, K):
        if root is None:
            return
        current_value = root.value
        label = root.label
        self.calculation_nums += 1
        distance = self.distance(target, current_value)

        is_duplicate = [self.distance(current_value, item[0].value) < 1e-4 and
                        abs(label-item[0].label) < 1e-4 for item in self.knn_result]

        if not np.array(is_duplicate, bool).any() or allow_duplicate:
            if len(self.knn_result) < K:
                self.knn_result.append((root, distance))
            elif distance < self.knn_result[0][1]:
                self.knn_result = self.knn_result[1:]+[(root, distance)]

        self.knn_result = sorted(self.knn_result, key=lambda x: -x[1])
        cuttint_dimmesion = root.depth % self.dimension_len
        if abs(target[cuttint_dimmesion] - current_value[cuttint_dimmesion]) < self.knn_result[0][1] or len(self.knn_result) < K:
            self.search_knn_recursive(root.left, target, K)
            self.search_knn_recursive(root.right, target, K)
        elif target[cuttint_dimmesion] < current_value[cuttint_dimmesion]:
            self.search_knn_recursive(root.left, target, K)
        else:
            self.search_knn_recursive(root.right, target, K)


def load_data(filename):
    data = load_svmlight_file(filename)
    return data[0].toarray(), data[1]


def test(train_data_file, test_data_file, K=3):
    train_data, train_labels = load_data(train_data_file)
    start_build = time.time()
    kd_tree = KDTree(train_data, train_labels)
    end_build = time.time()

    test_data, test_labels = load_data(test_data_file)
    search_total_time = 0
    calculation_nums = 0

    for index, target in enumerate(test_data):
        start_search = time.time()
        calculation_nums += kd_tree.search_knn(target, K)
        end_search = time.time()
        search_total_time += end_search - start_search

    print("Time of building KDtree:      ", end_build - start_build)
    print("Time of average searching:    ", search_total_time / len(test_data))
    print("Number of average calcuation: ", calculation_nums / len(test_data))


if __name__ == '__main__':
    print('Test for dataset of ijcnn1:')
    test("./datasets/ijcnn1.tr.bz2", "./datasets/ijcnn1-t-my")

    print('Test for dataset of mnist:')
    test("./datasets/mnist.scale.bz2", "./datasets/mnist.scale.t-my")

    print('Test for dataset of shuttle:')
    test("./datasets/shuttle.scale.tr", "./datasets/shuttle.scale.t-my")
