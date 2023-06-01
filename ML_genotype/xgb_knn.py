#!/usr/bin/env python

import csv
import sys
import random
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix

##################3 CONFIGURABLE VARIABLES ##################################

# Number of neighbors to use.
K_COEFFICIENT = 3

# weight function used in prediction. Possible values:
#   ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#   ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
WEIGHTS_TYPE = "distance"

############################################################################

class OutputPrinter:
    def __init__(self, filename):
        self.fp_ = open(filename, 'w')

    def __del__(self):
        self.fp_.close()

    def print(self, s):
        print (s)
        self.fp_.write(s + "\n")

class CsvInputFile:
    def __init__(self, filename):
        self.raw_rows_ = []
        self.data_ = dict()
        with open(filename, 'r') as data:
            reader = csv.reader(data)
            self.raw_rows_ = [row for row in reader]
        self.header_ = self.raw_rows_[0]
        for data_row in self.raw_rows_[1:]:
            cl = int(data_row[0])
            if not cl in self.data_:
                self.data_[cl] = []
            self.data_[cl].append([float(i) for i in data_row[1:]])

    def split_data(self, train_per_class = 6):
        train_x, train_y, test_x, test_y = [], [], [], []

        for cl, data in self.data_.items():
            if len(data) < train_per_class + 1:
                print ("Error: Class {0} doesn't have enough elements. Got {1}, but {2} needed at least." % 
                    (cl, len(data), train_per_class + 1))
                sys.exit(1)
            test_indexes = random.sample(range(len(data)), train_per_class)
            for i, row in enumerate(data):
                if i in test_indexes:
                    test_x.append(row)
                    test_y.append(cl)
                else:
                    train_x.append(row)
                    train_y.append(cl)
        return [(train_x, train_y), (test_x, test_y)]

# argparse function
def getargs():
    parser = argparse.ArgumentParser(prog='xgb_knn.py',
                            formatter_class=argparse.RawTextHelpFormatter,
                            description='')
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-i', '--infile', required=True,
                               help='input file name')
    requiredNamed.add_argument('-o', '--outfile', required=True,
                               help='output file name')
    return parser.parse_args()

def train_and_predict(train, test, active_features = []):
    knn = KNeighborsClassifier(n_neighbors=K_COEFFICIENT, weights=WEIGHTS_TYPE)
    train_x, train_y = [], train[1]
    for row in train[0]:
        filtered_row = []
        for i in range(len(row)):
            if len(active_features) > i and  active_features[i] == True:
                filtered_row.append(row[i])
        train_x.append(filtered_row)

    knn.fit(train_x, train_y)

    test_x, test_y = [], test[1]
    for row in test[0]:
        filtered_row = []
        for i in range(len(row)):
            if len(active_features) > i and active_features[i]:
                filtered_row.append(row[i])
        test_x.append(filtered_row)

    classes = sorted(set(train_y))
    predictions = []
    for i, row in enumerate(test_x):
        prediction = knn.predict([row])[0]
        predictions.append(prediction)
        probabilities = knn.predict_proba([row])[0]
        prob_str = ""
        for cli, p in enumerate(probabilities):
            if prob_str != "":
                prob_str += ", "
            prob_str += "%d: %.2f%c" % (classes[cli], p * 100.0, '%')

        printer.print ("Prediction for test sample #%d: %.0f, expected %d  (probabilities: %s)" % (i + 1, prediction, test_y[i], prob_str))
    printer.print ("Total prediction score (mean accuracy): %f" % knn.score(test_x, test_y))
    printer.print (classification_report(test_y, predictions))


#MAIN
if __name__ == "__main__":
    args = getargs()
    input_set = CsvInputFile(args.infile)
    printer = OutputPrinter(args.outfile)
    train, test = input_set.split_data()

    printer.print ("Training set: ")
    printer.print (','.join(input_set.header_))
    for i, row in enumerate(train[0]):
        printer.print ("%d,%s" % (train[1][i], ','.join([str(f) for f in train[0][i]])))

    printer.print ("Test set: ")
    printer.print (','.join(input_set.header_))
    for i, row in enumerate(test[0]):
        printer.print ("%d,%s" % (test[1][i], ','.join([str(f) for f in test[0][i]])))

    printer.print ("Analyzing features: ")

    for num_features in range(2, len(train[0][0]) + 1):
        knn = KNeighborsClassifier(n_neighbors=K_COEFFICIENT, weights=WEIGHTS_TYPE)
        xgb_estimator = XGBClassifier(use_label_encoder=False, verbosity=0, nthread=2)
        selector = RFE(estimator = xgb_estimator, n_features_to_select=num_features, step=1)
        # xgb requires int for y_train
        y_train = [int(x - 1) for x in train[1]]
        selector = selector.fit(train[0], y_train)


        printer.print ("-----------------------------------------------------------------------")
        printer.print ("Selecting top %d features." % num_features)
        top_features = []

        for i, is_top in enumerate(selector.support_):
            if is_top:
                top_features.append(input_set.header_[i+1])
        printer.print ("Top features (not sorted): %s" % ','.join(top_features))
        train_and_predict(train, test, selector.support_)

        other = [""] * (len(train[0][0]) - num_features)
        for i, r in enumerate(selector.ranking_):
            if r == 1: continue
            other[r - 2] = input_set.header_[i+1]
        if len(other) > 0:
            printer.print ("Ranking of other features (sorted): %s" % ','.join(other))
        elif len(other) == 0:
            printer.print ("Ranking of other features (sorted): .")
        printer.print ("-----------------------------------------------------------------------")
        printer.print ("")

