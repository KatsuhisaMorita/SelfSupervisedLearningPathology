# -*- coding: utf-8 -*-
"""
# Evaluation in Open TG-GATEs dataset
# finding/prognosis classification with KFold split (k=5)

@author: Katsuhisa MORITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVR
import lightgbm as lgb
from pulearn import BaggingPuClassifier, WeightedElkanotoPuClassifier
from evaluate import utils
import settings

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.size"] = 14

lst_classification=settings.lst_findings
lst_prognosis=settings.lst_findings
lst_compounds=settings.lst_compounds
lst_moa=settings.lst_moa
file_all=settings.file_all
file_classification=settings.file_classification
file_prognosis=settings.file_prognosis
file_moa=settings.file_moa

def load_conversion_dict(dir_data="", category="Category1"):
    # load conversion dict
    df=pd.read_csv(f"{dir_data}/240313_finding_table.csv")
    df=df.replace("-",np.nan).dropna(subset=["Open TG-GATEs",category])
    dict_name=dict(zip(df["Open TG-GATEs"],df[category]))
    return dict_name

class ClassificationFold:
    def __init__(self):
        self.df_info=None

    def evaluate(
        self,
        folder=f"",
        name="",
        layer=0,
        pretrained=False,
        n_fold=5,
        wsi=False, num_patch=None, strategy="max", random_state=24771,
        convertz=True,
        compression=False, n_components=16,
        pred_method="logistic_regression", params=dict(),
        finding=False, prognosis=False, #prediction mode/multi label
        moa=False, compound_name=False, #prediction mode/multi class
        lst_features=None,
        finding_base=False,
        delete_sample=False, lst_delete_conc=["Control", "Low"], #delete control/low conc samples with findings
        drop_injured_sample_train=False, 
        drop_injured_sample_test=False, 
        train_with_conversion_dict=None,
        eval_with_conversion_dict=None,
        file_classification=file_classification# for analysis
        ):
        # set
        dict_pred_method={
            "logistic_regression":self._pred_lr,
            "pu_learn_bagging":self._pred_pu_lr_bagging,
            "pu_learn_weightedelkanoto":self._pred_pu_lr_weightedelkanoto,
        }
        # load
        if finding:
            self._load_classification(
                filein=file_classification, 
                lst_features=lst_features, 
                train_with_conversion_dict=train_with_conversion_dict)
            if delete_sample:
                self._delete_sample(lst_delete_conc=lst_delete_conc)
        if prognosis:
            self._load_prognosis(lst_features=lst_features)
        if moa:
            pass # load for each fold
        if compound_name:
            pass # load for each fold

        # Predict / Evaluate
        lst_res=[]
        for fold in range(n_fold):
            if moa:
                self._load_moa(fold=fold, lst_features=lst_features)
            if compound_name:
                self._load_compound_name(fold=fold, lst_features=lst_features)
            if finding_base:
                arr_x_train, arr_x_test = self._load_labels(
                    self.df_info, fold=fold, convertz=convertz,
                    compression=compression, n_components=n_components,
                    prognosis=prognosis, constant=(pred_method=="constant"),
                )
            else:
                if wsi:
                    arr_x_train, arr_x_test = utils.load_array_fold_wsi(
                        self.df_info, fold=fold, layer=layer,
                        folder=folder, name=name, pretrained=pretrained,
                        num_patch=num_patch, strategy=strategy, random_state=random_state,
                        convertz=convertz,
                        compression=compression, n_components=n_components,
                    )
                else:
                    arr_x_train, arr_x_test = utils.load_array_fold(
                        self.df_info, fold=fold, layer=layer,
                        folder=folder, name=name, pretrained=pretrained,
                        convertz=convertz,
                        compression=compression, n_components=n_components,
                    )
            y_train = self.df_info.loc[self.df_info["FOLD"]!=fold, self.lst_features].values
            y_test = self.df_info.loc[self.df_info["FOLD"]==fold, self.lst_features].values
            if finding or prognosis:
                if pred_method=="constant":
                    y_pred=arr_x_test
                    res = utils.calc_stats_multilabel(y_test, y_pred, self.lst_features, eval_with_conversion_dict=eval_with_conversion_dict)
                else:
                    if drop_injured_sample_train or drop_injured_sample_test:
                        y_pred, y_test = self._predict_multilabel_drop(
                            arr_x_train, arr_x_test, y_train, y_test,
                            params, dict_pred_method[pred_method], train=drop_injured_sample_train, test=drop_injured_sample_test)
                    else:
                        y_pred = self._predict_multilabel(
                            arr_x_train, arr_x_test, y_train, 
                            params, dict_pred_method[pred_method])
                    res = utils.calc_stats_multilabel(
                        y_test, y_pred, self.lst_features, 
                        drop=drop_injured_sample_train or drop_injured_sample_test, 
                        eval_with_conversion_dict=eval_with_conversion_dict)
                    
            elif moa or compound_name:
                y_pred = self._predict_multiclass(
                    arr_x_train, arr_x_test, y_train, 
                    params) # only lr is implemented
                res = utils.calc_stats_multiclass(y_test, y_pred, )
            lst_res.append(res)
        return lst_res

    def _predict_multilabel(self, x_train, x_test, y_train, params, pred):
        """prediction with logistic regression for multi label task"""
        y_pred_all=[]
        # for loop for one feature
        for i in range(y_train.shape[1]):
            y_pred = pred(x_train, x_test, y_train[:,i], params)
            y_pred_all.append(y_pred)
        y_pred_all=np.stack(y_pred_all).T
        return y_pred_all

    def _predict_multilabel_drop(self, x_train, x_test, y_train, y_test, params, pred, train=False, test=False):
        """prediction with logistic regression for multi label task"""
        y_pred_all=[]
        y_test_all=[]
        # for loop for one feature
        for i in range(y_train.shape[1]):
            if train:
                ind_eval = (y_train[:,i]>0)|(y_train.sum(axis=1)==0)
                x_train_temp, y_train_temp=x_train[ind_eval], y_train[ind_eval,i]
            else:
                x_train_temp, y_train_temp=x_train, y_train[:,i]
            if test:
                ind_eval = (y_test[:,i]>0)|(y_test.sum(axis=1)==0)
                x_test_temp, y_test_temp=x_test[ind_eval], y_test[ind_eval,i]
            else:
                x_test_temp, y_test_temp=x_test, y_test[:,i]
            y_pred = pred(x_train_temp, x_test_temp, y_train_temp, params)
            y_pred_all.append(y_pred)
            y_test_all.append(y_test_temp)
        return y_pred_all, y_test_all

    def _predict_multiclass(self, x_train, x_test, y_train, params):
        """prediction with logistic regression for one label, multi class task"""
        lr = LogisticRegression(**params)
        lr.fit(x_train, y_train)
        y_pred=lr.predict_proba(x_test)
        return y_pred

    def _pred_lr(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_test)[:,1]
        return y_pred

    def _pred_pu_lr_bagging(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        pu_model = BaggingPuClassifier(
            estimator=model, n_estimators=15, n_jobs=-1, random_state=0
            )
        pu_model.fit(x_train, y_train)
        y_pred = pu_model.predict_proba(x_test)[:,1]  
        return y_pred

    def _pred_pu_lr_weightedelkanoto(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        pu_model = WeightedElkanotoPuClassifier(
            estimator=model, 
            labeled=20, unlabeled=10, 
            hold_out_ratio=0.2)
        pu_model.fit(x_train, y_train)
        y_pred = pu_model.predict_proba(x_test)[:,1]  
        return y_pred

    def _load_classification(
        self,
        filein=file_classification, 
        lst_features=None,
        train_with_conversion_dict=None,
        ):
        self.df_info=pd.read_csv(filein)
        self.df_info["INDEX"]=list(range(self.df_info.shape[0]))
        if train_with_conversion_dict:
            lst_key=[key for key in train_with_conversion_dict.keys()]
            lst_key=list(set(lst_key)&set(self.df_info.columns))
            df_key=self.df_info.loc[:,lst_key].T
            df_key["upper"]=[train_with_conversion_dict.get(i, i) for i in lst_key]
            df_key=df_key.groupby(by="upper",).max()
            self.lst_features=df_key.index.tolist()
            self.df_info=self.df_info.drop(lst_key, axis=1)
            self.df_info=pd.concat([self.df_info, df_key.T],axis=1)
        else:
            if lst_features:
                self.lst_features=lst_features
            else:
                self.lst_features=lst_classification

    def _delete_sample(
        self,
        lst_delete_conc=["Control", "Low"],
        ):
        s_tf=self.df_info.loc[:,self.lst_features].sum(axis=1)>0 #with at least one finding
        self.df_info=self.df_info[~((self.df_info["DOSE_LEVEL"].isin(lst_delete_conc))&(s_tf))] # drop true and concentration is (control or low sample)

    def _drop_injured_sample(
        self,
        arr_x_train, arr_x_test, y_train, y_test,
        fold:int=0, only_train=False,
        ):
        df_info_train=self.df_info[self.df_info["FOLD"]!=fold]
        df_info_train.index=df_info_train["INDEX"].tolist()
        df_info_test=self.df_info[self.df_info["FOLD"]!=fold]
        df_info_test.index=df_info_test["INDEX"].tolist()
        if only_train:
            df_temp=df_info_train[:,self.lst_features]

    def _load_prognosis(
        self,
        filein=file_prognosis, 
        lst_features=None):
        self.df_info=pd.read_csv(filein)
        if lst_features:
            self.lst_features=lst_features
        else:
            self.lst_features=lst_prognosis

    def _load_moa(
        self,
        fold:int=None,
        filein_all=file_all,
        filein_moa=file_moa,
        lst_features=None,
        ):
        if lst_features:
            lst_moa=lst_features
        # load metadata
        df_info = pd.read_csv(filein_all)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        moa_df = pd.read_csv(filein_moa).rename(columns={"Unnamed: 0":"COMPOUND_NAME"})
        df_info = pd.merge(df_info, moa_df, on = "COMPOUND_NAME")
        df_info = df_info[df_info["SACRI_PERIOD"].isin(["4 day", "8 day", "15 day", "29 day"]) & (df_info["DOSE"]>0)]
        # drop moa not existing moa in fold
        lst_tf=(df_info.loc[df_info["FOLD"]==fold,lst_moa].sum()!=0).tolist()
        lst_moa_fold=[lst_moa[v] for v, i in enumerate(lst_tf) if i]
        df_info["MoA"] = np.argmax(df_info.loc[:,lst_moa_fold].values,axis=1)
        # set
        self.df_info = df_info.loc[:,["MoA","FOLD","INDEX"]]
        self.lst_features="MoA"

    def _load_compound_name(
        self,
        fold:int=None,
        filein_all=file_all,
        filein_moa=file_moa,
        lst_features=None,
        ):
        if lst_features:
            lst_compounds=lst_features
        # load metadata
        df_info = pd.read_csv(filein_all)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        moa_df = pd.read_csv(filein_moa).rename(columns={"Unnamed: 0":"COMPOUND_NAME"})
        moa_df["MoA"] = np.argmax(moa_df[moa_df.columns[1:]].values,axis=1)
        df_info = pd.merge(df_info, moa_df, on = "COMPOUND_NAME")
        df_info = df_info[df_info["SACRI_PERIOD"].isin(["4 day", "8 day", "15 day", "29 day"]) & (df_info["DOSE"]>0)]
        # drop samples not existing compounds in fold
        lst_tf=(df_info.loc[df_info["FOLD"]==fold,lst_compounds].sum()!=0).tolist()
        lst_compounds_fold=[lst_compounds[v] for v, i in enumerate(lst_tf) if i]
        df_info=df_info[df_info["COMPOUND_NAME"].isin(lst_compounds_fold)]
        df_info["COMP"] = np.argmax(df_info.loc[:,lst_compounds_fold].values,axis=1)
        # set
        self.df_info = df_info.loc[:,["COMP","FOLD","INDEX"]]
        self.lst_features="COMP"

    def _load_labels(
        self,
        df_info, 
        filein=file_all,
        fold:int=0, 
        convertz=False,
        compression=False, n_components=128,
        prognosis=False, constant=False,
        ):
        ###finding label###
        df=pd.read_csv(filein)
        if prognosis:
            if constant:
                arr_x=df.loc[:,self.lst_features].values
            else:
                arr_x=df.loc[:,lst_prognosis].values
        else:
            arr_x=df.loc[:,lst_classification].values
        ind_train=df_info[df_info["FOLD"]!=fold]["INDEX"].tolist()
        ind_test=df_info[df_info["FOLD"]==fold]["INDEX"].tolist()
        arr_x_train, arr_x_test = arr_x[ind_train,:], arr_x[ind_test,:]
        if compression:
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
            arr_x_train, arr_x_test = utils.pca(arr_x_train, arr_x_test, n_components=n_components)
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
        elif convertz:
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
        return arr_x_train, arr_x_test
            
class PredFold:
    """For example, for biochemical values prediction"""
    def __init__(self):
        self.df_info=None

    def evaluate(
        self,
        folder=f"",
        name="",
        layer=0,
        pretrained=False,
        convertz=True,
        compression=False, n_components=16,
        pred_method="", params=dict(),
        lst_features=list(),
        ):
        # pred method
        dict_pred_method={
            "lenear_regression":self._pred_lr,
            "elasticnet":self._pred_els,
            "svm":self._pred_svm,
            "lgb":self._pred_lgb,
        }

        # load info
        self._load_info(lst_features=lst_features)

        # Predict / Evaluate
        lst_res=[]
        for fold in range(5):
            arr_x_train, arr_x_test = utils.load_array_fold(
                self.df_info, fold=fold, layer=layer,
                folder=folder, name=name, pretrained=pretrained,
                convertz=convertz,
                compression=compression, n_components=n_components,
            )
            y_train = self.df_info.loc[self.df_info["FOLD"]!=fold, self.lst_features].values
            y_test = self.df_info.loc[self.df_info["FOLD"]==fold, self.lst_features].values
            y_pred = self._pred(
                arr_x_train, arr_x_test, y_train,
                params, dict_pred_method[pred_method],
                )
            res = utils.calc_stats_regression(y_test, y_pred, self.lst_features)
            lst_res.append(res)
        return lst_res
    
    def _load_info(
        self,
        filein=file_all,
        lst_features=list()):
        self.df_info=pd.read_csv(filein)
        self.df_info["INDEX"]=list(range(self.df_info.shape[0]))
        self.df_info=self.df_info[self.df_info["COMPOUND_NAME"]!="propranolol"]# propranolol's rats contain abnormal biochamicel values
        self.df_info=self.df_info.loc[:,lst_features+["FOLD","INDEX"]]
        self.lst_features=lst_features

    def _pred(self, x_train, x_test, y_train, params, pred):
        """prediction with logistic regression for multi task"""
        y_pred_all=[]
        # for loop for one feature
        for i in range(y_train.shape[1]):
            y_pred = pred(x_train, x_test, y_train[:,i], params)
            y_pred_all.append(y_pred)
        y_pred_all=np.stack(y_pred_all).T
        return y_pred

    def _pred_lr(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).reshape(-1,1)
        return y_pred

    def _pred_els(self, x_train, x_test, y_train, params):
        model = ElasticNet(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred

    def _pred_svm(self, x_train, x_test, y_train, params):
        model=SVR(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred

    def _pred_lgb(self, x_train, x_test, y_train, params):
        train_dataset = lgb.Dataset(x_train, y_train)
        model = lgb.train(
            train_set = train_dataset, 
            params = params, 
            )
        y_pred = model.predict(x_test)
        return y_pred

class ClusteringFold:
    """Pseudo F Score"""
    def __init__(self):
        self.df_info=None

    def evaluate(
        self,
        folder="",
        name="",
        layer=0,
        pretrained=False,
        finding_base=False,
        convertz=True,
        compression=False, n_components=16,
        target="",
        random_f=True,
        random_state=0,
        ):
        # load info
        self._load_info(target=target)

        # Predict / Evaluate
        lst_f=[]
        lst_f_random=[]
        for fold in range(5):
            if finding_base:
                arr_x = self._load_labels(
                    self.df_info, fold=fold,
                    convertz=convertz,
                    compression=compression, n_components=n_components,
                )
            else:
                _, arr_x = utils.load_array_fold(
                    self.df_info, fold=fold, layer=layer,
                    folder=folder, name=name, pretrained=pretrained,
                    convertz=convertz,
                    compression=compression, n_components=n_components,
                )
            df_info_temp = self.df_info.loc[self.df_info["FOLD"]==fold]
            df_info_temp.loc[:,"INDEX"]=list(range(df_info_temp.shape[0]))
            if random_f:
                f_random=utils.pseudo_F(np.random.default_rng(random_state).permutation(arr_x,axis=0), df_info_temp, target)
                lst_f_random.append(f_random)
            f=utils.pseudo_F(arr_x, df_info_temp, target)
            lst_f.append(f)                
        if random_f:
            return lst_f, lst_f_random
        else:
            return lst_f
    
    def _load_info(
        self,
        filein_all=file_all,
        filein_moa=file_moa,
        target=""
        ):
        # load metadata
        df_info = pd.read_csv(filein_all)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        moa_df = pd.read_csv(filein_moa).rename(columns={"Unnamed: 0":"COMPOUND_NAME"})
        moa_df["MoA"] = np.argmax(moa_df[moa_df.columns[1:]].values,axis=1)
        df_info = pd.merge(df_info, moa_df, on = "COMPOUND_NAME")
        df_info = df_info[df_info["SACRI_PERIOD"].isin(["4 day", "8 day", "15 day", "29 day"]) & (df_info["DOSE"]>0)]
        # set
        self.df_info = df_info.loc[:,[target,"FOLD","INDEX"]]
        self.target=target

    def _load_labels(
        self,
        df_info, 
        filein=file_all,
        fold:int=0, 
        convertz=False,
        compression=False, n_components=16,
        ):
        ###finding label###
        df=pd.read_csv(filein)
        arr_x=df.loc[:,lst_classification].values
        ind_train=df_info[df_info["FOLD"]!=fold]["INDEX"].tolist()
        ind_test=df_info[df_info["FOLD"]==fold]["INDEX"].tolist()
        arr_x_train, arr_x_test = arr_x[ind_train,:], arr_x[ind_test,:]
        if compression:
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
            arr_x_train, arr_x_test = utils.pca(arr_x_train, arr_x_test, n_components=n_components)
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
        elif convertz:
            arr_x_train, arr_x_test = utils.standardize(arr_x_train, arr_x_test)
        return arr_x_test #only retrun test array