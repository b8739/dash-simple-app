""" """ """""" """""" """ 03. ANOMALY DETECTION  """ """""" """""" """" """

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np

## For Verification Data : TTA 테스트 데이터 (7개)
def create_isolation(X_train):
    isolation_forest = IsolationForest(
        n_estimators=500, max_samples=256, random_state=1
    )
    isolation_forest.fit(X_train)
    return isolation_forest


def anomaly_detect(df_veri, X_train):
    isolation_forest = create_isolation(X_train)

    X_veri = df_veri.drop(columns=["date", "Biogas_prod"])
    veri_y = df_veri["Biogas_prod"]

    a_scores_veri = -1 * isolation_forest.score_samples(X_veri)

    # print("verification 이상감지")
    # print(np.where(a_scores_veri >= 0.60))
    # print(a_scores_veri[np.where(a_scores_veri >= 0.60)])
