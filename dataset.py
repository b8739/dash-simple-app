# import pandas as pd

# # print(sys.path)
# import prepare_data
# import algorithm

# # monitoring page는 띄움과 동시에 df가 계산된다. cache된다.
# # modeling page는 띄움과 동시에 modeling이 계산된다. cache된다. 다시 하면 다시 계산되지 않는다.


# """ Dataset Preprocess """
# df = pd.read_csv("ketep_biogas_data_20220210.csv")
# df = prepare_data.preprocess(df)
# df_veri = prepare_data.extract_veri(df)
# train_Xn, train_y, test_Xn, test_y, X_test = prepare_data.split_dataset(df)
# print("dataset working")

# import math

# rep_prediction = {"value": math.inf}

# """ Modeling """
# for algorithm_type in ["xgb", "rf", "svr"]:
#     # 모델 만들고 실행
#     model = algorithm.create_model(algorithm_type, train_Xn, train_y)
#     result = algorithm.run(algorithm_type, model, test_Xn, test_y)
#     # 대푯값 비교해서 최소값으로 갱신
#     if rep_prediction["value"] > result["RMSE"]:
#         rep_prediction["algorithm"] = algorithm_type
#         rep_prediction["prediction"] = result["prediction"]

# # Actual: test_y
# # Predict: xgb_model_predict (rep_prediction['prediction])

# """ Actual Predictive Dataframe"""
# result_df = algorithm.get_actual_predictive(
#     X_test, test_y, rep_prediction["prediction"]
# )
# print(result_df)


# # xgb_model = algorithm.create_model('xgb', train_Xn, train_y)
# # result = algorithm.run('xgb', xgb_model, test_Xn, test_y)

# # svr_model = algorithm.create_model('svr', train_Xn, train_y)
# # algorithm.run('xgb', svr_model, test_Xn, test_y)

# # rf_model = algorithm.create_model('rf', train_Xn, train_y)
# # algorithm.run('xgb', rf_model, test_Xn, test_y)

# # 위 반복문에서 뭐가 대푯값이 될지 비교하는 코드가 하나 들어가야 함


# # get_actual_predictive(X_test, test_y, xgb_model_predict)

# # 3-1 Modeling예측값 대표로 하나 (아마 xgboost인데 뭘로 비교하는지 모름)

# # 3-2 예측값 실제값 그래프

# # - 예측값 실제값 코드에 어디부분인지 확인할것

# # 3-3 MAPE 와 RMSE

# # 3-4 (코드 없어서 아마 미정?)

# # 4-1 XGB 알고리즘 주요 변수 (only 수치)

# # 4-2 변수중요도 (bar graph)

# # 4-3 shap
