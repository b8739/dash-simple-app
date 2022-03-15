# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:40:14 2022

@author: s
"""

##### 2022-02-15 : BIOGAS PLANT MODEL #####


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)
import statsmodels.formula.api as sm
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, kstest
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, ward, complete, single

from xgboost import XGBRegressor, plot_importance
import shap


def MAPE(y, pred):
    return np.mean(np.abs((y - pred) / y) * 100)


os.chdir("C:/biogas/")


""" """ """""" """""" """ 00. PREPARING DATA SET  """ """""" """""" """" """

#### 01. OPEN DATA & PROCESSING ##

df_00 = pd.read_excel("ketep_biogas_data_20220314.xlsx")

df_00.drop(["Date_0"], axis=1, inplace=True)  # Delete 'Date_0' column

df_00.rename(
    columns={"Date": "date"}, inplace=True
)  # Change column name from 'Date' to 'date'

df_00.dtypes  # Find data types

df_00["date"] = pd.to_datetime(
    df_00["date"]
)  # Change data type from 'object' to 'datetime'
df_01 = df_00.iloc[0:1022, :]  # Train & Test data set

df_01.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs

df_veri = df_00.iloc[1022:1029, :]  # Data for Verifying (TTA Test)


## EXTRACT X & y SEPARATELY ##
X = df_01.drop("Biogas_prod", axis=1)  # Take All the columns except 'Biogas_prod'
y = df_01["Biogas_prod"]  # Take 'Biogas_prod' column
print(X.shape, y.shape)

## SET 'TRAIN', 'TEST' DATA, TRAIN/TEST RATIO, & 'WAY OF RANDOM SAMPLING' ##
X_train, X_test, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=12345
)
# X_train, X_test, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 56789)

train_x = X_train.drop(["date"], axis=1)  # Delete 'date' column from train data
test_x = X_test.drop(["date"], axis=1)  # Delete 'date' column from test data

# scalerX = MinMaxScaler()
scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)
# scalerX = RobustScaler()

scalerX.fit(train_x)
train_Xn = scalerX.transform(train_x)  # Scaling the train data
test_Xn = scalerX.transform(test_x)  # Scaling the test data

# train_b = scalerX.inverse_transform(train_Xn)


""" """ """""" """""" """ 01. MONITORING GRAPHS  """ """""" """""" """" """


def ts_plot_n2(data, col_name, n):  # Time-series graph using column name
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data = data[len(data[col_name]) - n : len(data[col_name])]
    plt.figure(figsize=(10, 4))
    plt.style.use("default")
    plt.title(col_name, fontsize=14)
    plt.plot(
        data["date"],
        data[col_name],
        color="darkorange",
        alpha=0.5,
        markersize=4.5,
        marker="o",
        linewidth=0.5,
    )  ## 'orange', 'steelblue'
    plt.ylim(
        [
            (data[col_name].min() - 3 * (data[col_name].std())),
            (data[col_name].max() + 3 * (data[col_name].std())),
        ]
    )
    axes = plt.gca()
    axes.yaxis.grid()
    return plt.show()


ts_plot_n2(df_01, "PS_feed_A", 60)
ts_plot_n2(df_01, "FW_Feed_A", 60)
ts_plot_n2(df_01, "Dig_A_Temp", 120)
ts_plot_n2(df_01, "PS_incoming", 120)
ts_plot_n2(df_01, "Dig_A_TS", 60)
ts_plot_n2(df_01, "Dig_Dewater", 60)


df_01.Biogas_prod.quantile(0.25)
df_01.Biogas_prod.quantile(0.5)
df_01.Biogas_prod.quantile(0.75)
df_01.Biogas_prod.describe()

df_01.columns


# v
# Out[138]:
# count    1016.000000
# mean       40.479823
# std         1.986788
# min        36.200000
# 25%        38.400000
# 50%        41.200000
# 75%        42.200000
# max        43.500000


""" """ """""" """""" """ 02. PREDICTIVE MODELING  """ """""" """""" """" """

""" """ """""" " 1. XGBOOST ALGORITHM  " """""" " " ""

xgb_model = xgb.XGBRegressor(
    n_estimators=1800,
    learning_rate=0.01,
    gamma=0.1,
    eta=0.04,
    subsample=0.75,
    colsample_bytree=0.5,
    max_depth=7,
)

xgb_model.fit(train_Xn, train_y)

xgb_model_predict = xgb_model.predict(test_Xn)


# CONFIRM PREDICTION POWER #
print("R_square_XGB :", r2_score(test_y, xgb_model_predict))
print("RMSE_XGB :", mean_squared_error(test_y, xgb_model_predict) ** 0.5)
print("MAPE_XGB :", MAPE(test_y, xgb_model_predict))


""" Draw performance graph : 'Actual' vs 'Predictive' """


def result_plot(x1, y_act, y_pred, point1, point2, width=12, height=5):
    z0 = pd.DataFrame(x1["date"])
    z0 = z0.reset_index(drop=True)
    z1 = pd.DataFrame(y_act)
    z1 = z1.reset_index(drop=True)
    z2 = pd.DataFrame(y_pred)
    result = pd.concat([z0, z1, z2], axis=1)
    result.columns = ["date", "Actual", "Predictive"]
    result = result.sort_values(by=["date"], axis=0, ascending=True)
    result = result.set_index("date")
    ## Graphs ##
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(width, height))
    plt.plot(
        result.iloc[point1:point2, 0],
        color="steelblue",
        alpha=0.5,
        markersize=4.5,
        marker="o",
        linewidth=0.5,
    )
    plt.plot(
        result.iloc[point1:point2, 1],
        color="darkorange",
        alpha=0.3,
        markersize=4.0,
        marker="o",
        linewidth=0.5,
    )
    plt.title("Actual(blue) vs. Predictive(orange)")
    plt.show()


### Plot for test data set
result_plot(X_test, test_y, xgb_model_predict, 0, 60, width=12, height=5)


""" Feature Importance """
# importance_type
# ‘weight’ - the number of times a feature is used to split the data across all trees.
# ‘gain’ - the average gain across all splits the feature is used in.
# ‘cover’ - the average coverage across all splits the feature is used in.
# ‘total_gain’ - the total gain across all splits the feature is used in.
# ‘total_cover’ - the total coverage across all splits the feature is used in.


feature_important = xgb_model.get_booster().get_score(importance_type="weight")
keys = list(feature_important.keys())
values = list(feature_important.values())

fimp_01 = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
    by="score", ascending=False
)
fimp_01.nlargest(30, columns="score").plot(
    kind="barh", figsize=(10, 8)
)  ## plot top 40 features


""" """ """""" " 2. RANDOM FOREST ALGORITHM  " """""" """ """

# #scalerX = MinMaxScaler()
# scalerX = StandardScaler()
# #scalerX = RobustScaler()

# scalerX.fit(train_x)
# train_Xn = scalerX.transform(train_x)
# test_Xn = scalerX.transform(test_x)


rf_model = RandomForestRegressor(n_estimators=400, min_samples_split=3)

rf_model.fit(train_Xn, train_y)
rf_model_predict = rf_model.predict(test_Xn)

print("R_square :", r2_score(test_y, rf_model_predict))
print("RMSE :", mean_squared_error(test_y, rf_model_predict) ** 0.5)
print("MAPE :", MAPE(test_y, rf_model_predict))


# Plot the Result (Random Forest)

result_plot(X_test, test_y, rf_model_predict, 0, 30, width=12, height=5)


""" Feature Importance """
features = train_x.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="steelblue", align="center")
plt.yticks(range(len(indices)), features[indices])
plt.xlabel("Relative Importance")


""" """ """""" " 3. SVR ALGORITHM  " """""" """ """

svr_model = SVR(
    kernel="rbf",
    C=100000,
    epsilon=0.9,
    gamma=0.0025,
    cache_size=200,
    coef0=0.0,
    degree=3,
    max_iter=-1,
    tol=0.0001,
)
svr_model.fit(train_Xn, train_y)
svr_model_predict = svr_model.predict(test_Xn)

print("RMSE :", mean_squared_error(test_y, svr_model_predict) ** 0.5)
print("MAPE :", MAPE(test_y, svr_model_predict))

result_plot(X_test, test_y, svr_model_predict, 0, 30, width=12, height=5)


""" """ """""" """""" """"" Verification """ """""" """""" """ """
df_veri.reset_index(drop=True, inplace=True)


veri_x = df_veri.drop(
    ["date", "Biogas_prod"], axis=1
)  # Take All the columns except 'Biogas_prod'
veri_y = df_veri["Biogas_prod"]

scalerX.fit(train_x)
veri_Xn = scalerX.transform(veri_x)  # Scaling the verifying data

j = 2
xgb_veri_predict = xgb_model.predict(
    veri_Xn[j, :].reshape(-1, 26)
)  # Apply xgb data to svr model
svr_veri_predict = svr_model.predict(
    veri_Xn[j, :].reshape(-1, 26)
)  # Apply veri data to svr model
rf_veri_predict = rf_model.predict(
    veri_Xn[j, :].reshape(-1, 26)
)  # Apply veri data to svr model


# Results
print("XGB_Pred = ", xgb_veri_predict)
print("RF_Pred = ", rf_veri_predict)
print("SVR_Pred = ", svr_veri_predict)

print("Actual = ", veri_y[j])


### SHAP IMPORTANCE

from xgboost import XGBRegressor, plot_importance
import shap


model = XGBRegressor()
model.fit(train_x, train_y)

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_x)

shap.force_plot(explainer.expected_value, shap_values[0, :], train_x.iloc[0, :])
# plt.show()

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, train_x)

shap.initjs()
# 총 13개 특성의 Shapley value를 절댓값 변환 후 각 특성마다 더함 -> np.argsort()는 작은 순서대로 정렬, 큰 순서대로 정렬하려면
# 앞에 마이너스(-) 기호를 붙임
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))

# 영향력 top 2 컬럼
for i in range(2):
    shap.dependence_plot(top_inds[i], shap_values, train_x)

shap.summary_plot(shap_values, train_x)

shap.summary_plot(shap_values, train_x, plot_type="bar", color="steelblue")


shap_values = shap.TreeExplainer(model).shap_values(train_x)
shap.summary_plot(shap_values, train_x, plot_type="bar")

shap.dependence_plot(top_inds[0], shap_values, train_x)

shap_interaction_values = explainer.shap_interaction_values(train_x)
shap.summary_plot(shap_interaction_values, train_x)

shap.dependence_plot(
    ("Dig_A_Temp", "FW_Feed_A"),
    shap_interaction_values,
    train_x,
    display_features=train_x,
)


### GET VARIABLE NAMES & FEATURE IMPORTANCE VALUES ###

feature_names = train_x.columns
rf_resultX = pd.DataFrame(shap_values, columns=feature_names)
vals = np.abs(rf_resultX.values).mean(0)

shap_importance = pd.DataFrame(
    list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
)

shap_importance.sort_values(
    by=["feature_importance_vals"], ascending=False, inplace=True
)
shap_importance.head(20)


shap_values

shap_interaction_values

shap.plots.bar(shap_values, train_x)

shap.plots.bar(shap_values)


### 5. GRAPHS


def test_result(x, y):
    from sklearn import linear_model

    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X_train, train_y)
    aaa = list(X_train.columns.values)
    x_test = x[aaa]
    y_hat = linear_regression.predict(x_test)
    mape = MAPE(y, y_hat)
    bb = pd.Series(y.index)
    test_yy = y.reset_index(drop=True)
    result_model = pd.concat([pd.Series(y.index), test_yy, pd.Series(y_hat)], axis=1)
    result_model.columns = ["ID", "Actual", "Pred"]
    return result_model


## Test Result for Test data ##
Test_Result = test_result(X_test, test_y)

print(Test_Result)
print("MAPE: ", round(MAPE(Test_Result["Actual"], Test_Result["Pred"]), 4))

from sklearn.metrics import r2_score

print("R-squared : ", round(r2_score(Test_Result["Actual"], Test_Result["Pred"]), 4))


## PREPARATION FOR GRAPH ##
z0 = pd.DataFrame(X_test["date"])
z0 = z0.reset_index(drop=True)
z1 = pd.DataFrame(test_y)
z1 = z1.reset_index(drop=True)
z2 = pd.DataFrame(xgb_model_predict)

result_xgb = pd.concat([z0, z1, z2], axis=1)
result_xgb.columns = ["date", "Actual", "Predictive"]
result_xgb.dtypes
result_xgb = result_xgb.sort_values(by=["date"], axis=0, ascending=True)
result_XGB = result_xgb.set_index("date")

plt.style.use("seaborn-whitegrid")
# plt.style.use('ggplot')
# plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.plot(
    result_XGB.iloc[0:200, 0],
    color="dodgerblue",
    alpha=0.6,
    markersize=4.5,
    marker="o",
    linewidth=0.5,
)
plt.plot(
    result_XGB.iloc[0:200, 1],
    color="darkorange",
    alpha=0.6,
    markersize=4.0,
    marker="o",
    linewidth=0.5,
)
plt.title("Actual(blue) vs. XGB Predictive(orange)")
plt.show()


""" """ """""" """""" """ 03. ANOMALY DETECTION  """ """""" """""" """" """

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


#### 01. OPEN DATA & PROCESSING ##

df_00 = pd.read_excel("C:/biogas/ketep_biogas_data_20220314.xlsx")  # Open Excel file
df_00.drop(["Date_0"], axis=1, inplace=True)  # Delete 'Date_0' column
df_00.rename(
    columns={"Date": "date"}, inplace=True
)  # Change column name from 'Date' to 'date'

df_00["date"] = pd.to_datetime(
    df_00["date"]
)  # Change data type from 'object' to 'datetime'
df_01 = df_00.iloc[0:1022, :]  # Train & Test data set
df_01.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs

df_veri = df_00.iloc[1022:1029, :]  # Data for Verifying (TTA Test)


X = df_01.drop(["Biogas_prod"], axis=1)
# X.dropna(axis=0, inplace=True)             # Delete entire rows which have the NAs
y = df_01["Biogas_prod"]


# np.random.seed(1)
# df = df_01.iloc[np.random.permutation(len(df_01))]

# df_ad = df_01[:900]
# df_y = df_01["Biogas_prod"]
# df_validate = df[900:]

X_train_0, X_test_0, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=123
)
# X_val, val_y = df_validate, df_validate["Biogas_prod"]


print("Shapes:\nX_train:%s\ntrain_y:%s\n" % (X_train_0.shape, train_y.shape))
print("X_test:%s\ntest_y:%s\n" % (X_test_0.shape, test_y.shape))
# print("x_val:%s\ny_val:%s\n" % (X_val.shape, y_val.shape))

X_train_0 = X_train_0.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)
X_test_0 = X_test_0.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)


X_train = X_train_0.drop(columns=["date"])
X_test = X_test_0.drop(columns=["date"])


###############   APPLY "ISOLATION FOREST ALGORITHM"     ######################

isolation_forest = IsolationForest(n_estimators=500, max_samples=256, random_state=1)
isolation_forest.fit(X_train)


# 학습데이터에 대한 Score 값 확인 #
a_scores_train = -1 * isolation_forest.score_samples(X_train)
print(a_scores_train)

plt.figure(figsize=(15, 5))
plt.hist(a_scores_train, bins=100)
plt.xlabel("Average Path Lengths", fontsize=14)
plt.ylabel("Number of Data Points", fontsize=14)
plt.show()


# Score 0.6 이상이면 Anomaly 라 정의 : 이상치 위치 확인 #
print(np.where(a_scores_train >= 0.60))
print(a_scores_train[np.where(a_scores_train >= 0.60)])

over_6 = list(np.where(a_scores_train >= 0.60)[0])
a_scores_train_1st = -1 * isolation_forest.score_samples(X_train.iloc[over_6, :])
a_scores_train_1st


## 테스트 데이터에 대한 확인 ##
a_scores_test = -1 * isolation_forest.score_samples(X_test)
print(a_scores_test)

## For Test Data

plt.figure(figsize=(15, 5))
plt.hist(a_scores_test, bins=100)
plt.xlabel("Average Path Lengths", fontsize=14)
plt.ylabel("Number of Data Points", fontsize=14)
plt.show()

print(np.where(a_scores_test >= 0.60))
print(a_scores_test[np.where(a_scores_test >= 0.60)])

over_6 = list(np.where(a_scores_test >= 0.60)[0])
a_scores_test_1st = -1 * isolation_forest.score_samples(X_test.iloc[over_6, :])
a_scores_test_1st


## For Verification Data : TTA 테스트 데이터 (7개)

X_veri = df_veri.drop(columns=["date", "Biogas_prod"])
veri_y = df_veri["Biogas_prod"]

a_scores_veri = -1 * isolation_forest.score_samples(X_veri)
print(a_scores_veri)

plt.figure(figsize=(15, 5))
plt.hist(a_scores_veri, bins=100)
plt.xlabel("Average Path Lengths", fontsize=14)
plt.ylabel("Number of Data Points", fontsize=14)
plt.show()

print(np.where(a_scores_veri >= 0.60))
print(a_scores_veri[np.where(a_scores_veri >= 0.60)])


## Verification Data :
# '2020-06-06 00:00:00',  '2020-06-07 00:00:00'


## 개별 변수의 이상 여부를 결정할 수 있는 방법

X_train.quantile(0.025)
X_train.iloc[
    479,
]
compare_train = pd.concat(
    [
        pd.Series(X_train.quantile(0.025)),
        pd.Series(
            X_train.iloc[
                479,
            ]
        ),
    ],
    axis=1,
)

print(compare_train)  # 첫 번째 컬럼값 (2.5% 미만)보다 작은 값이면 이상치로 적용


###################      END OF ANOMALY DETECTION        ##################


ts_plot_n2(df_01, "Treated_FW", 120)

sns.distplot(df_01["Biogas_prod"])

df_01.Biogas_prod.mean()
