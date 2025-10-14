import pandas as pd
import numpy as np
import statistics




df = pd.read_csv("Data/titanic/train.csv")

def complete_missing_values(df):

   embarked_mode = df['Embarked'].mode()[0]
   df["Embarked"] = df["Embarked"].fillna(embarked_mode)

   df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.mean()))

   return df

def prepare_data(df):

    df["FamilySize"] = df["SibSp"] + df["Parch"]

    df["IsAlone"] = np.where(df["FamilySize"] > 1, 0,1).astype(int)


    gender_map = {"female": 0, "male": 1}

    df["Gender"] = df["Sex"].map(gender_map)

    embarked_map = {"C": 0, "S": 1, "Q": 2}

    df["Embarked"] = df["Embarked"].map(embarked_map)

    df.drop(["Name","Ticket", "Cabin","PassengerId","Sex","SibSp","Parch"], axis=1, inplace=True)

    return df

df = complete_missing_values(df)
data = prepare_data(df)

def get_data():
    return data
