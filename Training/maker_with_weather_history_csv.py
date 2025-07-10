import sys
import os

from threading import Thread
import pandas as pd


path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(sys.argv[0])), 
    'Training', 'local_data', 'weatherHistory.csv')

def scale_value(val:float) -> int:
    val = str(val)[:10].replace('.', '') #Scales the cell value by converting it to an integer including max 10 decimal digits.
    
    while val[0] == '0':
        val = val[1:]
            
    return int(float(val))


def scale_values(df:pd.DataFrame) -> None:
    i=0
    while True:
        try:
            df.iloc[i, 0] = scale_value(df.iloc[i, 0])
            df.iloc[i, 1] = scale_value(df.iloc[i, 1])
            df.iloc[i, 2] = scale_value(df.iloc[i, 2])
            df.iloc[i, 3] = scale_value(df.iloc[i, 3])
            df.iloc[i, 4] = scale_value(df.iloc[i, 4])
            df.iloc[i, 5] = scale_value(df.iloc[i, 5])
            df.iloc[i, 6] = scale_value(df.iloc[i, 6])
            
            i+=1
            
        except IndexError:
            break


def label_string_cells(df:pd.DataFrame, addrs_summary:dict, precips_type:dict) -> None:
    val_index_summary = 1
    val_precips_type = 1
    
    i=0
    while True: #Converts all string labels to unique integers from the Summary and Precip Type columns.
        try:
            current_summary = df.iloc[i, 0]
            current_precips_type = df.iloc[i, 1]
                    
            try:
                addrs_summary[current_summary]
                
            except KeyError:
                addrs_summary[current_summary] = val_index_summary
                val_index_summary += 1
            
            try:
                precips_type[current_precips_type]
                
            except KeyError:
                precips_type[current_precips_type] = val_precips_type
                val_precips_type += 1
            
            df.iloc[i, 0] = addrs_summary[current_summary] #Converts the text label to numeric units.
            df.iloc[i, 1] = precips_type[current_precips_type] # ...
            
            i+=1
                
        except IndexError:
            break


def parse_dataframe() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)

    try:
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
    
    df = (df-df.mean()) / df.std() #Normalizes class values ​​for better results
    
    process_1 = Thread(target=label_string_cells, args=(string_cols, addrs_summary, precips_type))
    process_1.start()
    
    process_2 = Thread(target=scale_values, args=(df,))
    process_2.start()
    
    process_2.join()
    process_1.join()
    
    df = pd.concat([string_cols, df], axis=1) #Combines back the DataFrame with the string columns
    
    df = df[df.duplicated(subset=[coloum for coloum in df.columns], keep=False)] # Identify and drop unique classes
    df = df.convert_dtypes()
    df = df.astype(int)
    
    df.to_csv(path, index=False)
    
    print("New table example:")
    print(df.head())
    
    return df, addrs_summary