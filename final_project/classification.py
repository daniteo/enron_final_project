import sys
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


import warnings

SELECT_K_BEST_K = [3, 5, 10]
REDUCER_COMPONENTS = [1, 2, 3]
CLASS_WEIGHT = ['balanced', None]
DECISION_TREE_MIN_SPLIT = [2, 4, 6, 8]
DECISION_TREE_SPLITTER = ['best', 'random']
DECISION_TREE_MAX_FEATURE = ['sqrt', 'log2', None]
DECISION_TREE_CRITERION = ['gini', 'entropy']
SVM_C = [0.05, 0.1, 0.5, 1, 10, 10**2]
SVM_GAMMA = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
SVM_KERNEL = ['rbf', 'sigmoid', 'linear']


def decision_tree_classifier(dataset, feature_list):

    print "===== Decision Trees ====="

    pipe = get_pipeline(DecisionTreeClassifier(random_state=42))

    param_grid = {
#        'scaler': [StandardScaler()],
        'selector__k': SELECT_K_BEST_K,
        'reducer__n_components': REDUCER_COMPONENTS,
        'classifier__criterion': DECISION_TREE_CRITERION,
        'classifier__splitter': DECISION_TREE_SPLITTER,
        'classifier__min_samples_split': DECISION_TREE_MIN_SPLIT,
        'classifier__max_features': DECISION_TREE_MAX_FEATURE,
        'classifier__class_weight': CLASS_WEIGHT
    }

    clf, params = run_classifier(pipe, param_grid, dataset, feature_list)

    return clf, params


def naive_bayes_classifier(dataset, feature_list):

    print "===== Naive Bayes ====="

    pipe = get_pipeline(GaussianNB())

    param_grid = {
        'scaler': [StandardScaler()],
        'selector__k': SELECT_K_BEST_K,
        'reducer__n_components': REDUCER_COMPONENTS
    }

    clf, params = run_classifier(pipe, param_grid, dataset, feature_list)

    print params

    return clf, params


def svm_classifier(dataset, feature_list):

    print "===== SVM ====="

    pipe = get_pipeline(SVC(random_state=42))

    param_grid = {
        'scaler': [StandardScaler()],
        'selector__k': SELECT_K_BEST_K,
        'reducer__n_components': REDUCER_COMPONENTS,
        'classifier__C': SVM_C,
        'classifier__gamma': SVM_GAMMA,
        'classifier__class_weight': CLASS_WEIGHT,
        'classifier__kernel': SVM_KERNEL
    }

    clf, params = run_classifier(pipe, param_grid, dataset, feature_list)

    return clf, params



def get_pipeline(classifier):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', classifier)
    ])

    return pipe

def run_classifier(pipe, params, dataset, feature_list):
    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tree_grid = GridSearchCV(pipe, params, scoring='f1', cv=cv_split)

    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    tree_grid.fit(features, labels)

    test_classifier(tree_grid.best_estimator_, dataset, feature_list)

    return tree_grid.best_estimator_, tree_grid.best_params_


warnings.filterwarnings('ignore')

