import pandas as pd

df1 = pd.read_csv("../train_data.csv")

df2 = pd.read_csv("../data.csv")

df_new = pd.concat([df1, df2], ignore_index=True)

print(len(df1))
print(len(df2))
print(len(df_new))

df_new.to_csv("../Sandbox/train.csv", index=False)