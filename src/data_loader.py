import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def load_data(data_path):
    """
    Loads data from the JSON path, standardizes column names, and maps label.
    """
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_json(data_path)
    
    # Standardize columns to lowercase
    df.columns = df.columns.str.lower()
    
    # Map type/label
    if "label" not in df.columns:
        if "type" in df.columns:
            if df["type"].dtype == object:
                df["label"] = (df["type"].str.lower() == "phishing").astype(int)
            else:
                df["label"] = df["type"].astype(int)
        else:
            raise ValueError("Required column 'type' or 'label' not found in dataset.")
            
    df["label"] = df["label"].astype(int)
    return df

def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Splits the dataframe into stratified training and testing subsets.
    """
    print(f"Splitting dataset into stratified train and test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["label"]
    )
    return train_df, test_df
