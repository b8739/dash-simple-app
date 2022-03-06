import pandas as pd

""" Feature Importance """
# importance_type
# ‘weight’ - the number of times a feature is used to split the data across all trees.
# ‘gain’ - the average gain across all splits the feature is used in.
# ‘cover’ - the average coverage across all splits the feature is used in.
# ‘total_gain’ - the total gain across all splits the feature is used in.
# ‘total_cover’ - the total coverage across all splits the feature is used in.


# xgboost 모델이 필요
def get_feature_importance(xgb_model):
    feature_important = xgb_model.get_booster().get_score(importance_type="weight")
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    fimp_01 = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
        by="score", ascending=False
    )
    fimp_01.nlargest(30, columns="score").plot(
        kind="barh", figsize=(10, 8)
    )  ## plot top 40 features
