import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor


pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

def LM(train_features, train_targets ,test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Linear Regression Model
    '''
    skmodel = lm.LinearRegression().fit(train_features, train_targets)
    print('LR train', skmodel.score(train_features, train_targets))
    print('LR test', skmodel.score(test_features, test_targets))
    print('intercept = ', skmodel.intercept_, '\n', 'slope=', skmodel.coef_, '\n')
    train_predictions = skmodel.predict(train_features)
    test_predictions = skmodel.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha=0.4, color='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha=0.4, color='r')
    #plt.scatter(targets, targets, label='Original', alpha=0.3, color='navajowhite')
    # Plot the perfect prediction line
    xmin, xmax = plt.xlim()
    plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


def DTR(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Decision Tree Regression model
    '''
    score =[]
    for i in range(1, 10):
        decision_tree1 = DecisionTreeRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('DTR best depth', np.argmax(score)+1)
    decision_tree = DecisionTreeRegressor(max_depth=np.argmax(score)+1)
    decision_tree.fit(train_features, train_targets)
    print('DTR train', decision_tree.score(train_features, train_targets))
    print('DTR test', decision_tree.score(test_features, test_targets), '\n')

    plt.figure(figsize=(35, 25))
    #tree.plot_tree(decision_tree, filled=True, rounded=True, feature_names=feature_names)
    plt.show()
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    # Scatter the predictions vs actual values
    plt.scatter(train_predictions, train_targets, label='train', alpha = 0.6, c='b')
    plt.scatter(test_predictions, test_targets, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Decision Tree')
    plt.legend()
    plt.show()


def RFR(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Random Forest Model
    '''
    score = []
    for i in range(1, 10):
        decision_tree1 = RandomForestRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('RF best depth', np.argmax(score)+1)
    rfr = RandomForestRegressor(n_estimators=400,
                                max_depth=np.argmax(score)+1,
                                random_state=42)
    rfr.fit(train_features, train_targets)
    # Look at the R^2 scores on train and test
    print('RF train', rfr.score(train_features, train_targets))
    print('RF test',rfr.score(test_features, test_targets), '\n')
    train_predictions = rfr.predict(train_features)
    test_predictions = rfr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Random Forest')
    plt.legend()
    plt.show()


def GBR(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Gradient Boosting Model
    '''
    score = []
    for i in range(1, 10):
        score = []
        decision_tree1 = GradientBoostingRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('GBR best depth', np.argmax(score)+1)
    gbr = GradientBoostingRegressor(n_estimators=400,
                                    random_state=42,
                                    max_depth=np.argmax(score)+1
                                    )
    gbr.fit(train_features, train_targets)
    print('GBR train', gbr.score(train_features, train_targets))
    print('GBR test', gbr.score(test_features, test_targets), '\n')
    train_predictions = gbr.predict(train_features)
    test_predictions = gbr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Gradient Boosting')
    plt.legend()
    plt.show()

def LoR(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Logistic Regression Model
    '''

    LoR = LogisticRegression()
    LoR.fit(train_features, train_targets)
    print('LoR train', LoR.score(train_features, train_targets))
    print('LoR test', LoR.score(test_features, test_targets), '\n')
    train_predictions = LoR.predict(train_features)
    test_predictions = LoR.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Logistic Regression')
    plt.legend()
    plt.show()

def SVC(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Support Vector Machine Model
    '''
    svc = svm.SVC(kernel='rbf')
    svc.fit(train_features, train_targets)
    print('Svc train', svc.score(train_features, train_targets))
    print('Svc test', svc.score(test_features, test_targets), '\n')
    train_predictions = svc.predict(train_features)
    test_predictions = svc.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha=0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha=0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('SVC')
    plt.legend()
    plt.show()

def XGBC(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Support Vector Machine Model
    '''
    model = XGBClassifier()
    model.fit(train_features, train_targets,eval_metric='logloss')
    print('XGBC train', model.score(train_features, train_targets))
    print('XGBC test', model.score(test_features, test_targets), '\n')
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha=0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha=0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('XGBoost')
    plt.legend()
    plt.show()

def DTC(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Decision Tree Regression model
    '''
    score =[]
    for i in range(1, 10):
        decision_tree1 = DecisionTreeClassifier(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('DTC best depth', np.argmax(score)+1)
    decision_tree = DecisionTreeClassifier(max_depth=np.argmax(score)+1)
    decision_tree.fit(train_features, train_targets)
    print('DTC train', decision_tree.score(train_features, train_targets))
    print('DTC test', decision_tree.score(test_features, test_targets), '\n')

    plt.figure(figsize=(35, 25))
    #tree.plot_tree(decision_tree, filled=True, rounded=True, feature_names=feature_names)
    plt.show()
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    # Scatter the predictions vs actual values
    plt.scatter(train_predictions, train_targets, label='train', alpha = 0.6, c='b')
    plt.scatter(test_predictions, test_targets, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Decision Tree')
    plt.legend()
    plt.show()

def RFC(train_features, train_targets, test_features, test_targets):
    '''

    :param train_features: Train Features
    :param train_targets: Train Targets
    :param test_features: Test Features
    :param test_targets: Test Targets
    :return: Random Forest Model
    '''
    score = []
    for i in range(1, 10):
        decision_tree1 = RandomForestClassifier(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('RFC best depth', np.argmax(score)+1)
    rfc = RandomForestClassifier(n_estimators=400,
                                max_depth=np.argmax(score)+1,
                                random_state=42)
    rfc.fit(train_features, train_targets)
    # Look at the R^2 scores on train and test
    print('RFC train', rfc.score(train_features, train_targets))
    print('RFC test',rfc.score(test_features, test_targets), '\n')
    train_predictions = rfc.predict(train_features)
    test_predictions = rfc.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Random Forest')
    plt.legend()
    plt.show()

