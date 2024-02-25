import time # for benchmarking
import numpy as np # to optimize the calculations for arrays
from matplotlib import pyplot as plt # to visualize the data
from collections import Counter # to count the most common class


def euclidean_distance(p1, p2):
    """
    To calculate the Euclidean distance between 2 points that have two dimensions:
    $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$
    """
    return np.sqrt(np.sum(np.power((np.array(p1) - np.array(p2)), 2)))


class knn():
    def __init__(self, k=3):
        self.k = k
        self.point = None

    def train(self, points):
        """
        There is no training in KNN per se, only storing the points happens.
        """
        self.points = points

    def predict(self, new_point):
        distances = [] # to keep track of all distances

        # get the distance from the point to all other data points
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(new_point, point)
                distances.append([distance, category])

        # get the categories of the first 'k' instances
        categories = [category[1] for category in sorted(distances)[0:self.k]]

        # the most common category is what we want
        return Counter(categories).most_common(1)[0][0]


def main():
    points = {"red": [[1,3], [2,4], [5,6]],
              "blue": [[10,9], [8,9]]}

    new_point = [4,5]

    start_time = time.time()
    # perform the KNN algorithm via a classifier
    classifier = knn()
    classifier.train(points)
    end_time = time.time()
    print(f"The prediction category: {classifier.predict(new_point)}")
    print(f"Time taken for original code: {end_time - start_time:.6f} seconds\n")


if __name__ == "__main__":
    main()
