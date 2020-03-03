import numpy as np
from math import sqrt
from math import exp
from math import pi

class GaussianNaiveBayes():
    def __init__(self):
        print("Custom GNB")
        self.mean = []
        self.std = []
        self.classes = []
        self.class_dict = {}
        self.stats = {}
        self.prob = {}

    def predict(self, x_train, y_train, x_test):
        y_pred = []
        classes = []
        for i in y_train:
            if i not in classes:
                classes.append(i)
        self.classes = classes
        for i in classes:
            self.class_dict[i] = []
            self.stats[i] = []
            self.prob[i] = 1
        for i in self.classes:
            for j in range(len(x_train)):
                if y_train[j] == i:
                    self.class_dict[i].append(x_train[j])
        for class_val,data in self.class_dict.items():
            for col in zip(*data):
                self.stats[class_val].append((np.mean(col), np.std(col)))
        if x_test is None:
            return None
        for row in x_test:
            for i in self.classes:
                self.prob[i] = 1
            for class_val, data in self.stats.items():
                for i in range(len(row)):
                    mean, std = data[i]
                    x = row[i]
                    self.prob[class_val] *= self._gaussian_probabilities_distribution(x, mean, std)
            #print(self.prob," for row ",row)
            minimum = 0
            cl = 0
            for classes, data in self.prob.items():
                if data > minimum:
                    minimum  = data
                    cl = classes
            y_pred.append(cl)
        return y_pred
        
        
    def _gaussian_probabilities_distribution(self, num, mean, standard_deviation):
        exponent = exp(-((num - mean) ** 2 / (2 * standard_deviation ** 2)))
        prob = (1/ (sqrt(2 * pi) * standard_deviation)) * exponent
        return prob