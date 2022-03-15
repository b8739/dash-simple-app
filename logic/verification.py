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
