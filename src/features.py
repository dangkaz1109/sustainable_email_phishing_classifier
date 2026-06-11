import re
import pandas as pd

def get_primary_text_column(df):
    """
    Identifies the primary text column (body, subject, or html) and fills NaNs.
    """
    text_column = None
    if "body" in df.columns:
        text_column = "body"
    elif "subject" in df.columns:
        text_column = "subject"
    elif "html" in df.columns:
        text_column = "html"
        
    if not text_column:
        raise ValueError("No text column (body, subject, or html) found in dataset.")
        
    df[text_column] = df[text_column].fillna("")
    if "subject" in df.columns:
        df["subject"] = df["subject"].fillna("")
        
    return text_column

def engineer_numeric_features(df, text_column):
    """
    Engineers and returns numeric features from email text fields.
    """
    df_copy = df.copy()
    
    # Feature engineering
    df_copy["subject_len"] = df_copy["subject"].apply(len) if "subject" in df_copy.columns else 0
    df_copy["body_len"] = df_copy[text_column].apply(len)
    
    df_copy["subject_word_count"] = df_copy["subject"].apply(lambda x: len(x.split())) if "subject" in df_copy.columns else 0
    df_copy["body_word_count"] = df_copy[text_column].apply(lambda x: len(x.split()))
    
    df_copy["body_digit_count"] = df_copy[text_column].apply(lambda x: sum(c.isdigit() for c in x))
    df_copy["body_url_count"] = df_copy[text_column].apply(lambda x: len(re.findall(r"https?://\S+|www\.\S+", x)))
    df_copy["body_special_char_count"] = df_copy[text_column].apply(lambda x: len(re.findall(r"[!@#$%^&*(),.?\":{}|<>]", x)))
    
    return df_copy
