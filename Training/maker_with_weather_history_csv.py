import os
import sys
import pandas as pd
import zipfile

from threading import Thread
from Training.Features.utils import scale_values, label_string_cells


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
    precips_type = {}
    
    for column in df.columns:    
        df.drop(df[df[column].isin(['null', None])].index, inplace=True) #Remove invalid values ​​from DataFrame
        
    df.reset_index(drop=True, inplace=True)
    
    string_cols = df.select_dtypes(include=['object', 'string'])
    df = df.select_dtypes(include=['number'])
        
    process_1 = Thread(target=label_string_cells, args=(string_cols, addrs_summary, precips_type))
    process_1.start()
    
    process_2 = Thread(target=scale_values, args=(df,))
    process_2.start()
    
    process_2.join()
    process_1.join()
        
    df = pd.concat([string_cols, df], axis=1) #Combines back the DataFrame with the old string columns as labels
    df = df.convert_dtypes().astype(int)
    
    df.to_csv(path, index=False)
    
    print("New normalized dataframe:")
    print(df.head())
    
    return df, {str(value): key for key, value in addrs_summary.items()}