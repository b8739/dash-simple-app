import shap
import pandas as pd
from xgboost import XGBRegressor, plot_importance
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import linear_model


def MAPE(y, pred):
    return np.mean(np.abs((y - pred) / y) * 100)


def verifyModel(
    xgb_model,
    svr_model,
    rf_model,
    df_veri,
    train_x,
):

    """ """ """""" """""" """"" Verification """ """""" """""" """ """
    df_veri.reset_index(drop=True, inplace=True)

    veri_x = df_veri.drop(
        ["date", "Biogas_prod"], axis=1
    )  # Take All the columns except 'Biogas_prod'
    veri_y = df_veri["Biogas_prod"]

    scalerX = StandardScaler()  # Data standardization (to Standard Normal distribution)

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


def get_shap_importance(train_x, train_y):
    ### SHAP IMPORTANCE

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

    def test_result(x, y, X_test, X_train, test_y):
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X_train, train_y)
        aaa = list(X_train.columns.values)
        x_test = x[aaa]
        y_hat = linear_regression.predict(x_test)
        mape = MAPE(y, y_hat)
        bb = pd.Series(y.index)
        test_yy = y.reset_index(drop=True)
        result_model = pd.concat(
            [pd.Series(y.index), test_yy, pd.Series(y_hat)], axis=1
        )
        result_model.columns = ["ID", "Actual", "Pred"]
        return result_model

        ## Test Result for Test data ##
        Test_Result = test_result(X_test, test_y)

        print(Test_Result)
        print("MAPE: ", round(MAPE(Test_Result["Actual"], Test_Result["Pred"]), 4))

        print(
            "R-squared : ",
            round(r2_score(Test_Result["Actual"], Test_Result["Pred"]), 4),
        )
