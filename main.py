import sys
import os
import argparse

import pandas as pd
import json

from threading import Thread

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1500)


path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(sys.argv[0])), 
    'Training', 'local_data')

def tuple_type(arg):
    try:
        values = [int(item) if item.isdigit() else item for item in arg.split(',')]
        return tuple(values)
    
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format. Must be comma-separated integers/strings without spaces.")
    

def to_train() -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score
    
    from Training.Features.utils import normalize_df
    
    df, labels = maker.parse_dataframe(path)
    
    print('\nNew labels for summary for prediction:')
    
    with open(os.path.join(path, 'labels.json'), "w") as file:
        json.dump(labels, file, indent=4)
    
    for name, label in labels.items():
        print(f"{label}: {name}")
    
    print('\nClose the plot window to continue...')
    plot_new_df(normalize_df(df))
    
    df = df.iloc[-48:] #Using the last 48 hours of history for weather forecasting
    
    X = df.drop('Summary', axis=1)
    y = df['Summary']
    
    '''vc = y.value_counts()
    mask = y.isin(vc[vc >= 2].index) #Identify and drop unique classes

    X = X[mask]
    y = y[mask]'''
    
    if len(y.unique()) > 1:
        print(f"\nSpecific number of unique classes for y: {len(y.unique())}")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\nPrediction labels: {y_pred}")
    
    print(f"\nModel trained successfully. Score: {accuracy_score(y_test, y_pred) * 100} %\n")
    
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0) * 100:.1f} %")
    print(f"Mean Squared Error: {mean_absolute_error(y_test, y_pred) * 100:.1f} %")
    print(f"Recall Score: {recall_score(y_test, y_pred, average='macro') * 100:.1f} %")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro') * 100:.1f} %")
    
    return labels
    
    
def plot_new_df(df:pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    
    normalized_df = df.copy().head()
    normalized_df.plot()
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Basic weather forecast model.')
    
    parser.add_argument('-t0', '--train_0', action='store_true', help='Train the model with the first type of dataset.')
    parser.add_argument('-t1', '--train_1', action='store_true', help='Train the model with the second type of dataset.')
    
    parser.add_argument('-p', '--predict', action='store_true', help='API-compatible address.')
    
    args = parser.parse_args()
    
    labels:dict = None
    
    if args.train_0:
        from Training import maker_with_weather_history_csv as maker
        
        labels = to_train()
        
    if args.train_1:
        
        
        labels = to_train() #.....Any script to parse data for the model.
        
    if args.predict:
        from Training.Features.utils import scale_values, label_string_cells
        import json
        
        print("\nSimulating a Prediction weather conditions...")    
        
        df = pd.DataFrame(
                {
                    'Summary': ["Clear"], #The correct answer will not be included in the prediction
                    'Precip Type': ["rain"],
                    'Temperature (C)': [13.844444444444445],
                    'Apparent Temperature (C)': [13.844444444444445],
                    'Humidity': [0.84],
                    'Wind Speed (km/h)': [0.0],
                    'Wind Bearing (degrees)': [0.0],
                    'Visibility (km)': [15.826300000000002],
                    'Pressure (millibars)': [1017.82]
                }
            )
        
        addrs_summary = {}
        precips_type = {}
        
        string_cols = df.select_dtypes(include=['object', 'string'])
        df = df.select_dtypes(include=['number'])
        
        process_1 = Thread(target=label_string_cells, args=(string_cols, addrs_summary, precips_type))
        process_1.start()
        
        process_2 = Thread(target=scale_values, args=(df,))
        process_2.start()
        
        process_2.join()
        process_1.join()
        
        df = pd.concat([string_cols, df], axis=1) #Combines back the DataFrame with the old string columns as labels
                
        df = df.drop('Summary', axis=1).convert_dtypes().astype(int) #The correct answer will not be included in the prediction
        
        if not labels:
            try:
                with open(os.path.join(path, 'labels.json'), "r") as file:
                    labels = json.load(file)
                    
            except FileNotFoundError:
                print("Labels file not found. Please train the model first.")
                sys.exit(1)

        print()
        print(df)
        print(f"\nPrediction label: {labels[clf.predict(df)[0]]}")