""" 
Max Plooling and Logistic Regression for MIL
"""

import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def standardize(x_train, x_test=None, train_only=False):
    """ standardize / fillna with 0 """
    ss = StandardScaler()
    if train_only:
        x_train = ss.fit_transform(x_train)
        x_train[np.isnan(x_train)]=0
        return x_train
    else:
        ss.fit(x_train)
        x_train = ss.transform(x_train)
        x_train[np.isnan(x_train)]=0
        x_test = ss.transform(x_test)
        x_test[np.isnan(x_test)]=0
        return x_train, x_test

class PoolingMIL:
    def __init__(self, strategy="max", random_state:int=24771):
        self.data=[]
        self.train_data=None
        self.test_data=None
        dict_strategy={
            "max":self._max_pooling,
            "min":self._min_pooling,
            "mean":self._mean_pooling,
        }
        self._pooling=dict_strategy[strategy]
        self.logistic_model=None
        random.seed(random_state)

    def pooling_data(self, X, num_patch:int=None):
        self.data.append(self._pooling(X, num_patch=num_patch))

    def set_train_data(self, X=[], ):
        if len(X)==0:
            self.train_data=np.stack(self.data)
            self.data=[]
        else:
            self.train_data=X

    def set_test_data(self, X=[], ):
        if len(X)==0:
            self.test_data=np.stack(self.data)
            self.data=[]
        else:
            self.test_data=X

    def save_model(self, folder=None, name=None, ):
        if name:
            temp=f"{folder}/{name}_"
        else:
            temp=f"{folder}/"
        np.save(f"{temp}coef.npy", self.logistic_model.coef_)
        np.save(f"{temp}intercept.npy", self.logistic_model.intercept_)

    def predict(self, x_train=None, x_test=None, y_train=None, params=dict(), convertz=False,):
        if not x_train:
            x_train=self.train_data
        if not x_test:
            x_test=self.test_data
        if convertz:
            x_train, x_test = standardize(x_train, x_test)
        self._train(x_train, y_train, params)
        if x_test.shape[0]!=0:
            return self._predict(x_test)

    def _train(self, x_train=None, y_train=None, params=dict(), ):
        self.logistic_model=LogisticRegression(**params)
        self.logistic_model.fit(x_train, y_train)

    def _predict(self, x_test=None):
        return self.logistic_model.predict_proba(x_test)[:,1]

    def _max_pooling(self, X, num_patch:int=None):
        if num_patch:
            lst_loc=random.sample(list(range(X.shape[0])), num_patch)
            return np.max(X[lst_loc], axis=0)
        else:
            return np.max(X, axis=0)

    def _min_pooling(self, X, num_patch:int=None):
        if num_patch:
            lst_loc=random.sample(list(range(X.shape[0])), num_patch)
            return np.min(X[lst_loc], axis=0)
        else:
            return np.min(X, axis=0)

    def _mean_pooling(self, X, num_patch:int=None):
        if num_patch:
            lst_loc=random.sample(list(range(X.shape[0])), num_patch)
            return np.mean(X[lst_loc], axis=0)
        else:
            return np.mean(X, axis=0)
        
    def _delete_data(self):
        self.train_data=[]
        self.test_data=[]
