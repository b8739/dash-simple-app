## Verification 2022-04-01





df_veri.reset_index(drop=True, inplace=True)



veri_x = df_veri.drop(['date', 'Biogas_prod'], axis = 1)  # Take All the columns except 'Biogas_prod'

veri_y = df_veri['Biogas_prod']    



scalerX.fit(train_x)

veri_Xn = scalerX.transform(veri_x)      # Scaling the verifying data





j=1



xgb_veri_predict = xgb_model.predict(veri_Xn[(j-1), :].reshape(-1, 26))  # Apply xgb data to svr model

svr_veri_predict = svr_model.predict(veri_Xn[(j-1), :].reshape(-1, 26))  # Apply veri data to svr model

rf_veri_predict = rf_model.predict(veri_Xn[(j-1), :].reshape(-1, 26))  # Apply veri data to svr model



# Results

print('XGB_Pred = ', xgb_veri_predict)

print('RF_Pred = ', rf_veri_predict)

print('SVR_Pred = ', svr_veri_predict)



print('Actual = ', veri_y[j-1])





## Training Model



# Updating Training Data

train_Xn_new = np.vstack([train_Xn, veri_Xn[(j-1)]])

train_y_new = pd.concat([train_y, pd.Series(veri_y[j-1])])





# XGB

xgb_model.fit(train_Xn_new, train_y_new)

xgb_model_predict = xgb_model.predict(test_Xn)



# CONFIRM PREDICTION POWER #

print('R_square_XGB :', r2_score(test_y, xgb_model_predict))

print('RMSE_XGB :', mean_squared_error(test_y, xgb_model_predict)**0.5)

print('MAPE_XGB :', MAPE(test_y, xgb_model_predict))



# RF

rf_model.fit(train_Xn, train_y)

rf_model_predict = rf_model.predict(test_Xn)



print('R_square :', r2_score(test_y, rf_model_predict))

print('RMSE :', mean_squared_error(test_y, rf_model_predict)**0.5)

print('MAPE :', MAPE(test_y, rf_model_predict))



# SVR

svr_model.fit(train_Xn, train_y)

svr_model_predict = svr_model.predict(test_Xn)



print('RMSE :', mean_squared_error(test_y, svr_model_predict)**0.5)

print('MAPE :', MAPE(test_y, svr_model_predict))

