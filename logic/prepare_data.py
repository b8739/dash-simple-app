import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


""" 원본 데이터프레임 (df_00) 받아서 데이터 가공하고 df_01 RETURN"""


def preprocess(df_00):
    #### 01. OPEN DATA & PROCESSING ##
    # 데이터가 읽어져서 패러미터로 들어오니 밑줄 주석
    # df_00 = pd.read_excel("C:/biogas/ketep_biogas_data_20220210.xlsx")  # Open Excel file
    # df_00.drop(["Date_0"], axis=1, inplace=True)  # Delete 'Date_0' column

    df_00.rename(
        columns={"Date": "date"}, inplace=True
    )  # Change column name from 'Date' to 'date'

    df_00.dtypes  # Find data types

    df_00["date"] = pd.to_datetime(
        df_00["date"]
    )  # Change data type from 'object' to 'datetime'
    df_01 = df_00.iloc[0:1022, :].copy()  # Train & Test data set

    df_01.dropna(axis=0, inplace=True)  # Delete entire rows which have the NAs
    return df_01


""" 원본 데이터프레임 (df_00) 받아서 df_veri RETURN"""


def extract_veri(df_00):
    df_veri = df_00.iloc[1022:1029, :].copy()  # Data for Verifying (TTA Test)
    return df_veri


""" 가공된 데이터 데이터프레임 (df_01) 받아서 X,y RETURN"""


def get_xy(df_01):
    ## EXTRACT X & y SEPARATELY ##
    X = df_01.drop("Biogas_prod", axis=1)  # Take All the columns except 'Biogas_prod'
    y = df_01["Biogas_prod"]  # Take 'Biogas_prod' column
    print(X.shape, y.shape)
    return X, y


""" 위 함수에서 x,y 받아서 train_Xn,train_y,test_Xn,test_y,X_test RETURN"""


def split_dataset(df_01):
    X, y = get_xy(df_01)
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
    return train_Xn, train_y, test_Xn, test_y, X_test
