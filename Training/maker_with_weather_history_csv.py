import os
import sys
import pandas as pd
import zipfile

from sklearn.preprocessing import LabelEncoder


def parse_dataframe(path:str) -> tuple[pd.DataFrame, dict]:
    folder = path
    
    labels_path = os.path.join(folder, 'labels.json')
    path = os.path.join(folder, 'weatherHistory.csv')
    zip_path = os.path.join(folder, 'weatherHistory.csv.zip')
    
    if os.path.exists(path):
        os.remove(path)
        
    if os.path.exists(labels_path):
        os.remove(labels_path)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)
            
        print(f"Successfully extracted '{zip_path}' to '{path}'.\n")
        
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file.")
        sys.exit()
        
    except FileNotFoundError:
        print(f"Error: Zip file '{zip_path}' not found.")
        sys.exit()
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()
    
    print("Parsing dataframe...\n")
    
    df = pd.read_csv(os.path.join(path))

    try: #Remove redundant and unnecessary information for training
        df.drop('Formatted Date', axis=1, inplace=True)
        df.drop('Daily Summary', axis=1, inplace=True)
        df.drop('Loud Cover', axis=1, inplace=True)
    except:
        pass
    
    addrs_summary = {}
    for index, value in df['Summary'].items():
        addrs_summary[value] = index
        
    for column in df.columns:    
        df.drop(df[df[column].isin(['null', None])].index, inplace=True) #Remove invalid values ​​from DataFrame
        
    df.reset_index(drop=True, inplace=True)
    
    df['Summary'] = LabelEncoder().fit_transform(df['Summary'])
    df['Precip Type'] = LabelEncoder().fit_transform(df['Precip Type'])
    
    df.to_csv(path, index=False)
    
    print("New fixed dataframe:")
    print(df.head())
        
    _addrs_summary = {}
    for key, value in addrs_summary.items():
        _addrs_summary[str(df['Summary'][value])] = key
            
    return df, _addrs_summary