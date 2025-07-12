import pandas as pd


def normalize_df(df:pd.DataFrame) -> pd.DataFrame:
    return (df-df.mean()) / df.std()


def scale_value(val:float) -> int:
    return int(val * 1000)


def scale_values(df:pd.DataFrame) -> pd.DataFrame:
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
        
    return df


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
            
            df.iloc[i, 0] = addrs_summary[current_summary] #Converts the string label to numeric units.
            df.iloc[i, 1] = precips_type[current_precips_type] # ...
            
            i+=1
                
        except IndexError:
            break