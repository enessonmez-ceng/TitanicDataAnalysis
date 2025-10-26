import pandas as pd
import numpy as np
import statistics


df = pd.read_csv("Data/titanic/train.csv")

def handle_outliers_iqr(df, column):
    """
    Bir sütundaki aykırı değerleri IQR yöntemine göre baskılar (capping).
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"'{column}' için Aykırı Değer Sınırları: Alt={lower_bound:.2f}, Üst={upper_bound:.2f}")

  
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


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

    df.drop(["Name","Ticket", "Cabin","PassengerId","Sex","SibSp","Parch","IsAlone"], axis=1, inplace=True)

    return df


df = complete_missing_values(df)

df = handle_outliers_iqr(df, 'Fare')
df = handle_outliers_iqr(df, 'Age')

data = prepare_data(df)

def get_data():
    return data
