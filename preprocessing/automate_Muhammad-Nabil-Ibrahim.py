import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    columns_for_boxplotting=['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    for index in range(len(columns_for_boxplotting)):
        Q1=df[columns_for_boxplotting[index]].quantile(0.25)
        Q3=df[columns_for_boxplotting[index]].quantile(0.75)

        IQR=Q3-Q1

        lower_limit=Q1-1.5*IQR
        upper_limit=Q3+1.5*IQR

        df[columns_for_boxplotting[index]]=df[columns_for_boxplotting[index]].clip(lower=lower_limit,upper=upper_limit)

    df=pd.concat([df,pd.get_dummies(df['Thallium'],prefix='Thallium',drop_first=True,dtype=int)],axis=1)
    df=df.drop(columns=['Thallium'])

    df=pd.concat([df,pd.get_dummies(df['Chest pain type'],prefix='Chest pain type',drop_first=True,dtype=int)],axis=1)
    df=df.drop(columns=['Chest pain type'])

    df=pd.concat([df,pd.get_dummies(df['EKG results'],prefix='EKG results',drop_first=True,dtype=int)],axis=1)
    df=df.drop(columns=['EKG results'])

    df=pd.concat([df,pd.get_dummies(df['Slope of ST'],prefix='Slope of ST',drop_first=True,dtype=int)],axis=1)
    df=df.drop(columns=['Slope of ST'])

    df=pd.concat([df,pd.get_dummies(df['Number of vessels fluro'],prefix='Number of vessels fluro',drop_first=True,dtype=int)],axis=1)
    df=df.drop(columns=['Number of vessels fluro'])

    columns_for_scaling=['Age','BP','Cholesterol','Max HR','ST depression']
    scaler=StandardScaler()
    df[columns_for_scaling]=scaler.fit_transform(df[columns_for_scaling])

    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def run():
    df = load_data("../heart-dissease-dataset/Heart_Disease_Prediction.csv")
    df_clean = preprocess_data(df)
    save_data(
        df_clean,
        "Eksperimen_SML_Muhammad-nabil-ibrahim/preprocessing/data_clean.csv"
    )

if __name__ == "__main__":
    run()
