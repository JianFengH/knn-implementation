import numpy as np
import time
from sklearn.datasets import load_svmlight_file

allow_duplicate = False


class Ball():
    def __init__(self, center, radius, points, left, right):
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points


class BallTree():
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels
        if(len(self.values) != len(self.labels)):
            raise ValueError("values and labels must be the same length")
        if(len(self.values) == 0):
            raise ValueError("values must be non-empty")
        
        data = np.column_stack((self.values, self.labels))
        self.root = self.build_BallTree(data)
        self.knn_max_now_dist = np.inf
        self.knn_result = [(None, self.knn_max_now_dist)]

    def distance(self, point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))

    def build_BallTree(self, data):
        if len(data) == 0:
            return None
        if len(data) == 1:
            return Ball(data[0, :-1], 0.001, data, None, None)
        data_disloc = np.row_stack((data[1:], data[0]))
        if np.sum(data_disloc-data) == 0:
            return Ball(data[0, :-1], 1e-100, data, None, None)
        cur_center = np.mean(data[:, :-1], axis=0)
        dists_with_center = np.array(
            [self.distance(cur_center, point) for point in data[:, :-1]])
        max_dist_index = np.argmax(dists_with_center)
        max_dist = dists_with_center[max_dist_index]
        root = Ball(cur_center, max_dist, data, None, None)
        point1 = data[max_dist_index]

        dists_with_point1 = np.array(
            [self.distance(point1[:-1], point) for point in data[:, :-1]])
        max_dist_index2 = np.argmax(dists_with_point1)
        point2 = data[max_dist_index2]

        dists_with_point2 = np.array(
            [self.distance(point2[:-1], point) for point in data[:, :-1]])
        assign_point1 = dists_with_point1 < dists_with_point2

        root.left = self.build_BallTree(data[assign_point1])
        root.right = self.build_BallTree(data[~assign_point1])
        return root

    def search_knn(self, target, K):
        if self.root is None:
            raise ValueError("tree is empty")
        if K > len(self.values):
            raise ValueError("K in knn Must Be Greater Than Lenght of data")
        if len(target) != len(self.root.center):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.knn_result = [(None, self.knn_max_now_dist)]
        self.nums = 0
        self.search_knn_recursive(self.root, target, K)
        return self.nums

    def insert(self, root_ball, target, K):
        for node in root_ball.points:
            self.nums += 1
            is_duplicate = [self.distance(node[:-1], item[0][:-1]) < 1e-4 and
                            abs(node[-1] - item[0][-1]) < 1e-4 for item in self.knn_result if item[0] is not None]
            if np.array(is_duplicate, bool).any() and not allow_duplicate:
                continue
            distance = self.distance(target, node[:-1])
            if(len(self.knn_result) < K):
                self.knn_result.append((node, distance))
            elif distance < self.knn_result[0][1]:
                self.knn_result = self.knn_result[1:] + [(node, distance)]
            self.knn_result = sorted(self.knn_result, key=lambda x: -x[1])

    def search_knn_recursive(self, root_ball, target, K):
        if root_ball is None:
            return
        if root_ball.left is None or root_ball.right is None:
            self.insert(root_ball, target, K)
        if abs(self.distance(root_ball.center, target)) <= root_ball.radius + self.knn_result[0][1]:
            self.search_knn_recursive(root_ball.left, target, K)
            self.search_knn_recursive(root_ball.right, target, K)

def load_data(filename):
    data = load_svmlight_file(filename)
    return data[0].toarray(), data[1]

def test(train_data_file, test_data_file, K=3):
    train_data, train_labels = load_data(train_data_file)
    start_build = time.time()
    ball_tree = BallTree(train_data, train_labels)
    end_build = time.time()

    test_data, test_labels = load_data(test_data_file)
    search_total_time = 0
    calculation_nums = 0

    for index, target in enumerate(test_data):
        start_search = time.time()
        calculation_nums += ball_tree.search_knn(target, K)
        end_search = time.time()
        search_total_time += end_search - start_search

    print("Time of building BallTree:    ", end_build - start_build)
    print("Time of average searching:    ", search_total_time / len(test_data))
    print("Number of average calcuation: ", calculation_nums / len(test_data))


if __name__ == '__main__':
    print('Test for dataset of ijcnn1:')
    test("./datasets/ijcnn1.tr.bz2", "./datasets/ijcnn1-t-my")

    print('Test for dataset of mnist:')
    test("./datasets/mnist.scale.bz2", "./datasets/mnist.scale.t-my")

    print('Test for dataset of shuttle:')
    test("./datasets/shuttle.scale.tr", "./datasets/shuttle.scale.t-my")
