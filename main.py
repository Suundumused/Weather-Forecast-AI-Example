import argparse
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


def to_train() -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    df, labels = maker.parse_dataframe()
    
    print('\nNew labels for summary for prediction:')
    
    for name, label in labels.items():
        print(f"{label}: {name}")
        
    plot_new_df(df)
    
    X = df.drop('Summary', axis=1)
    y = df['Summary']
    
    if len(y.unique()) > 1:
        print(f"\nNumber of unique classes: {len(y.unique())}")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\nPrediction labels: {y_pred}")
    print(f"\nModel trained successfully. Score: {accuracy_score(y_test, y_pred) * 100} %")
    
    return labels
    
    
def plot_new_df(df:pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    
    normalized_df = df.copy().head()
    normalized_df.plot()
    
    plt.show()
    
    
def predict(df:pd.DataFrame, cloud_address:str) -> pd.DataFrame:
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Basic weather forecast model.')
    
    parser.add_argument('-t0', '--train_0', action='store_true', help='Train the model with the first type of dataset.')
    parser.add_argument('-t1', '--train_1', action='store_true', help='Train the model with the second type of dataset.')
    
    parser.add_argument('-p', '--predict', type=str, default='https://any...', help='Cloud-compatible address.')
    
    args = parser.parse_args()
    
    if args.train_0:
        from Training import maker_with_weather_history_csv as maker
        
        labels = to_train()
        
    if args.train_1:
        #.....Any script to parse data for the model.
        
        labels = to_train()
        
    if args.predict:
        pass