from config import DATA
import pandas as pd
from datasets import Dataset

def load_dataframe():
    return pd.read_csv(DATA, sep= ";")

def load_my_dataset():
    return Dataset.load_dataset(DATA)
