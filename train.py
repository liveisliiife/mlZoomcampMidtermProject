# libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle


# parameters
number_leaf=300
depth = 8
output_file = "model_midterm.bin"
model_file = "model_midterm.bin"

# data preparation

df = pd.read_csv("heart_attack_prediction_dataset.csv")

df.drop("Patient ID",axis=1,inplace=True)  

df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic'] = df['Systolic'].astype(int)
df['Diastolic'] = df['Diastolic'].astype(int)
df.drop("Blood Pressure",axis=1,inplace=True) 


df.columns = df.columns.str.replace(" ","_").str.lower()

categorical_columns = list(df.dtypes[df.dtypes=="object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ","_")


df.heart_attack_risk = (df.heart_attack_risk == 1).astype(int)

numerical = []
categorical = []

for col in df.columns:
    if df[col].dtypes == "object":
        categorical.append(col)
    else:
        numerical.append(col)
numerical.remove("heart_attack_risk")



df = df.drop(["sex","diet","hemisphere","continent"],axis=1)
categorical = ["country"]

df = df.drop(["previous_heart_problems","bmi"],axis=1)
numerical.remove("previous_heart_problems")
numerical.remove("bmi")





df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=1)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)


y_train = df_train.heart_attack_risk.values
y_val = df_val.heart_attack_risk.values
y_test = df_test.heart_attack_risk.values


del df_train["heart_attack_risk"]
del df_val["heart_attack_risk"]
del df_test["heart_attack_risk"]


dv = DictVectorizer(sparse=False)


train_dicts = df_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient="records")
X_val = dv.transform(val_dicts) 


# decisio tree


dt = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=number_leaf)
dt.fit(X_train,y_train)
y_pred = dt.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val,y_pred)


# Final Model

dv_full = DictVectorizer(sparse=False)

dict_full_train = df_full_train.to_dict(orient="records")

X_full_train = dv_full.fit_transform(dict_full_train)

y_full_train = df_full_train.heart_attack_risk.values

model = DecisionTreeClassifier(max_depth=8,min_samples_leaf=300)

model.fit(X_full_train,y_full_train)

dicts_test = df_test.to_dict(orient="records")

X_test = dv_full.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:,1] 
roc_auc_score(y_test, y_pred)
print(f"auc{auc}")


# model deployment 


# model saving

with open(output_file,"wb") as f_out:   # write,binary
    pickle.dump((dv_full,model),f_out)