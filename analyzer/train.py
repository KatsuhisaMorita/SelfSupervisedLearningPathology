# -*- coding: utf-8 -*-
"""
# train out-fold0 dataset of Open TG-GATEs
# pu-laern, logistic regression models for finding classification
# modified from evaluate/tggatefold.py

@author: Katsuhisa MORITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVR
from pulearn import BaggingPuClassifier, WeightedElkanotoPuClassifier
from evaluate import utils
import settings

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 14

lst_classification = settings.lst_findings
file_classification = settings.file_classification


def load_conversion_dict(dir_data="", category="Category1"):
    # load conversion dict
    df = pd.read_csv(f"{dir_data}/240313_finding_table.csv")
    df = df.replace("-", np.nan).dropna(subset=["Open TG-GATEs", category])
    dict_name = dict(zip(df["Open TG-GATEs"], df[category]))
    return dict_name


class ClassificationModel:
    def __init__(self):
        self.df_info = None

    def train(
        self,
        folder=f"",
        name="layer",
        layer=45,
        fold=0,
        pretrained=False,
        wsi=False,
        num_patch=None,
        strategy="max",
        random_state=24771,
        convertz=True,
        compression=False,
        n_components=16,
        pred_method="logistic_regression",
        params=dict(),
        lst_features=None,
        train_with_conversion_dict=None,
        eval_with_conversion_dict=None,
        file_classification=file_classification,
    ):
        dict_pred_method = {
            "logistic_regression": self._pred_lr,
            "pu_learn_bagging": self._pred_pu_lr_bagging,
            "pu_learn_weightedelkanoto": self._pred_pu_lr_weightedelkanoto,
        }
        self._load_classification(
            filein=file_classification,
            lst_features=lst_features,
            train_with_conversion_dict=train_with_conversion_dict,
        )

        if wsi:
            arr_x_train, arr_x_test = utils.load_array_fold_wsi(
                self.df_info,
                fold=fold,
                layer=layer,
                folder=folder,
                name=name,
                pretrained=pretrained,
                num_patch=num_patch,
                strategy=strategy,
                random_state=random_state,
                convertz=convertz,
                compression=compression,
                n_components=n_components,
            )
        else:
            arr_x_train, arr_x_test = utils.load_array_fold(
                self.df_info,
                fold=fold,
                layer=layer,
                folder=folder,
                name=name,
                pretrained=pretrained,
                convertz=convertz,
                compression=compression,
                n_components=n_components,
            )
        y_train = self.df_info.loc[
            self.df_info["FOLD"] != fold, self.lst_features
        ].values
        y_test = self.df_info.loc[
            self.df_info["FOLD"] == fold, self.lst_features
        ].values
        lst_models, y_pred = self._predict_multilabel(
            arr_x_train, arr_x_test, y_train, params, dict_pred_method[pred_method]
        )
        # evaluation
        self.result = utils.calc_stats_multilabel(
            y_test,
            y_pred,
            self.lst_features,
            drop=False,
            eval_with_conversion_dict=eval_with_conversion_dict,
        )
        self.lst_models = lst_models

    def save_models(
        self,
        outdir: str = "",
        style="each",
        savename="models_dict",
    ):
        """
        save models
        outdir:str
        style:str each or dict (one file)
        """
        if style == "each":
            for name, model in zip(self.lst_features, self.lst_models):
                pd.to_pickle(
                    model,
                    f"{outdir}/{name}.pickle",
                )
                print(f"saved: {outdir}/{name}.pickle")
        elif style == "dict":
            pd.to_pickle(
                dict(zip(self.lst_features, self.lst_models)),
                f"{outdir}/{savename}.pickle",
            )
            print(f"saved: {outdir}/{savename}.pickle")

    def _predict_multilabel(self, x_train, x_test, y_train, params, pred):
        """prediction with logistic regression for multi label task"""
        y_pred_all = []
        lst_models = []
        # for loop for one feature
        for i in range(y_train.shape[1]):
            model, y_pred = pred(x_train, x_test, y_train[:, i], params)
            y_pred_all.append(y_pred)
            lst_models.append(model)
        y_pred_all = np.stack(y_pred_all).T
        return lst_models, y_pred_all

    def _pred_lr(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_test)[:, 1]
        return model, y_pred

    def _pred_pu_lr_bagging(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        pu_model = BaggingPuClassifier(
            estimator=model,
            n_estimators=15,
            n_jobs=-1,
        )
        pu_model.fit(x_train, y_train)
        y_pred = pu_model.predict_proba(x_test)[:, 1]
        return pu_model, y_pred

    def _pred_pu_lr_weightedelkanoto(self, x_train, x_test, y_train, params):
        model = LogisticRegression(**params)
        pu_model = WeightedElkanotoPuClassifier(
            estimator=model, labeled=20, unlabeled=10, hold_out_ratio=0.2
        )
        pu_model.fit(x_train, y_train)
        y_pred = pu_model.predict_proba(x_test)[:, 1]
        return pu_model, y_pred

    def _load_classification(
        self,
        filein=file_classification,
        lst_features=None,
        train_with_conversion_dict=None,
    ):
        self.df_info = pd.read_csv(filein)
        self.df_info["INDEX"] = list(range(self.df_info.shape[0]))
        if train_with_conversion_dict:
            lst_key = [key for key in train_with_conversion_dict.keys()]
            lst_key = list(set(lst_key) & set(self.df_info.columns))
            df_key = self.df_info.loc[:, lst_key].T
            df_key["upper"] = [train_with_conversion_dict.get(i, i) for i in lst_key]
            df_key = df_key.groupby(
                by="upper",
            ).max()
            self.lst_features = df_key.index.tolist()
            self.df_info = self.df_info.drop(lst_key, axis=1)
            self.df_info = pd.concat([self.df_info, df_key.T], axis=1)
        else:
            if lst_features:
                self.lst_features = lst_features
            else:
                self.lst_features = lst_classification
