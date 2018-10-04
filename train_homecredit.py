# https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
# HomeCreditコンペで、多くの人のベースとなったカーネルを参考に作成
# http://kurupical.hatenablog.com/entry/2018/09/10/221420　の３特徴量だけを使うと同じ
import numpy as np
import pandas as pd
import gc
import time
import os
from datetime import datetime
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


train = pd.read_feather("./input/application_train.ftr")
test = pd.read_feather("./input/application_test.ftr")

# EXT_SOURCEという外部の信用機関の情報３つに絞る
train = train.loc[:, ['SK_ID_CURR', 'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
test = test.loc[:, ['SK_ID_CURR', 'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
print("Train shape: {}, test shape: {}".format(train.shape, test.shape))


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(train, test, num_folds=3, stratified=False):
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # log 書き込み用
    list_cv_score = []
    list_cv_best_iteration = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train['TARGET'])):
        train_x, train_y = train[feats].iloc[train_idx], train['TARGET'].iloc[train_idx]
        valid_x, valid_y = train[feats].iloc[valid_idx], train['TARGET'].iloc[valid_idx]

        # https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
        clf = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.10,
            num_leaves=30,  # main parameter
            colsample_bytree=0.8197036,
            max_bin=100,
            max_depth=-1,
            reg_alpha=1,
            reg_lambda=3,
            min_split_gain=0.0222415,
            min_child_weight=0,
            min_child_samples=70,
            silent=True,
            # 以下はGPUの指定する場合
            # device='gpu',
            # gpu_platform_id=0,
            # gpu_device_id=0
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=10, early_stopping_rounds=40)

        # predict_probaを使うようにしている
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        # ログに書き込むようにリストの末尾に加える
        list_cv_score.append(roc_auc_score(valid_y, oof_preds[valid_idx]))
        list_cv_best_iteration.append(clf.best_iteration_)

        del train_x, train_y, valid_x, valid_y  # clf, は消さないことにした
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train['TARGET'], oof_preds))
    with open('output/cv_log.text', 'a', encoding='utf8')as output:
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=output)
        print("dtrain:", train[feats].iloc[train_idx].shape, "      dtest:",
              train[feats].iloc[valid_idx].shape, file=output)
        print(clf.get_params, file=output)
        print("score,std", np.mean(list_cv_score), np.std(list_cv_score), list_cv_score, file=output)
        print("best_iteration:  ", np.mean(list_cv_best_iteration), np.std(list_cv_best_iteration),
              list_cv_best_iteration, file=output)

    del clf
    gc.collect()

    # Write submission file and plot feature importance
    test['TARGET'] = sub_preds
    test[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)  # score 0.71235

    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().\
           sort_values(by="importance", ascending=False)[:20].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 4))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('output/lgbm_importances01.png')


def main():
    with timer("Run LightGBM with kfold"):
        # feat_importance = kfold_lightgbm(df, num_folds= 2, stratified= False, debug= debug)
        feature_importance_df = kfold_lightgbm(train, test, num_folds=3, stratified=False,)

        with open('output/cv_log.text', 'a', encoding='utf8')as output:
            print("", file=output)
            print("--------------", file=output)
    display_importances(feature_importance_df)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    submission_file_name = "output/submission_homecredit.csv"
    with timer("Full model run"):
        main()
