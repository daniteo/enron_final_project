#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Inicialmente selecionaremos todos as caracteristicas (exceto email, que e apenas informativa)
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'deferred_income',
                 'deferral_payments',
                 'loan_advances',
                 'other',
                 'expenses',
                 'director_fees',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'from_messages',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
import data_handling
### Aqui serao removidas os outliers conforme analise feita no Jupyter no arquivo UdacityFinalProject.ipynb
data_dict = data_handling.outlier_cleaning(data_dict)
### Aqui sao feitos ajustes nos dados com valores divergentes
data_dict = data_handling.data_handling(data_dict)

### Task 3: Create new feature(s)
import features_selection as fs

data_dict, new_financial_features = fs.create_new_financial_features(data_dict)
data_dict, new_email_feaures = fs.create_poi_email_ratio_features(data_dict)

features_list_2 = features_list + new_financial_features + new_email_feaures

best_features_and_score = fs.select_k_best(data_dict, features_list_2, 10)

best_features = ['poi']

for feature in best_features_and_score:
    print "Feature: {:25} - Score: {}".format(feature[0], feature[1])
    best_features.append(feature[0])

### Store to my_dataset for easy export below.
my_dataset = data_dict
my_feature_data = best_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_data, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier

data_dict = data_handling.remove_nan(data_dict)
my_dataset = data_dict

# Create and test the Gaussian Naive Bayes Classifier with all features
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)

clf = GaussianNB()
test_classifier(clf, my_dataset, best_features)
# Create and test the Decision Tree with all features
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, best_features)
# Create and test the SVM with all features
clf = SVC()
test_classifier(clf, my_dataset, best_features)
# Create and test the KMEans  with all features
clf = KMeans(n_clusters=2)
test_classifier(clf, my_dataset, best_features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

import classification as cl


clf, _ = cl.naive_bayes_classifier(my_dataset, my_feature_data)

### Para avaliar os demais algoritimos basta remover os comentários.
#clf, _ = cl.svm_classifier(my_dataset, my_feature_data)
#clf, _ = cl.decision_tree_classifier(my_dataset, my_feature_data)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_data)