from config import PATH_TO_DATA
import pandas as pd
#from datasets import Dataset

def load_dataframe():
    return pd.read_csv(PATH_TO_DATA, sep= ";")

# def load_my_dataset():
#     return Dataset.load_dataset(DATA)
