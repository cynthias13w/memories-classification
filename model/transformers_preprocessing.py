# installing packages
# !pip install transforms datasets
# !pip install evaluate
# !pip install accelerate -U
# !pip install transformers[torch]
from transformers import DistilBertTokenizerFast
# 1. Load Data
#from datasets import Dataset
from data import load_dataframe

# Firstly load dataframe
data = load_dataframe()

# load Dataset
dataset = Dataset.from_pandas(data[['story', 'memType']])
dataset = dataset.rename_column('memType', 'labels')
dataset = dataset.rename_column('story', 'text')

# id2label = {0: "imagined", 1: "recalled", 2:"retold"}
# label2id = {"imagined":0, "recalled": 1, "retold":2}

# 2. Preprocessing
def tokenizer_function(dataset):
    """
    Args:
        dataset: Sequence of string to tokenize
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer(dataset['text'], padding='max_length', truncation=True)

def tokenize_data(data):
    return data.map(tokenizer_function, batched = True)
