#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def check_outliers(features):

    ### read in data dictionary, convert to numpy array
    data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
    data = featureFormat(data_dict, features)

    ### your code below

    max_y = 0
    outlier = ""

    for item in data_dict:
        y = data_dict[item][features[1]]
        x = data_dict[item][features[0]]
        if y != 'NaN':
            if y > max_y:
                max_y = y
                outlier = item
#        if x != 'NaN' and y != 'NaN':
#            print item
    print outlier, max_y

    plot_data(data, features)

    data_dict.pop(outlier, 0)
#    features = ["salary", "bonus"]
    data = featureFormat(data_dict, features)
    plot_data(data, features)


def plot_data(data, labels):

    for point in data:
        x_axis = point[0]
        y_axis = point[1]
        matplotlib.pyplot.scatter(x_axis, y_axis)

    matplotlib.pyplot.xlabel(labels[0])
    matplotlib.pyplot.ylabel(labels[1])
    matplotlib.pyplot.show()

if __name__ == "__main__":
    check_outliers(["salary", "bonus"])