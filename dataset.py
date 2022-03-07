import pandas as pd

# print(sys.path)
import prepare_data

df = pd.read_csv("ketep_biogas_data_20220210.csv")
df = prepare_data.preprocess(df)
df_veri = prepare_data.extract_veri(df)
train_Xn, train_y, test_Xn, test_y, X_test = prepare_data.split_dataset(df)
