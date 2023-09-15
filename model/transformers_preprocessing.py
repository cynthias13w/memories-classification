from transformers import AutoTokenizer, DataCollatorWithPadding
#DistilBertTokenizerFast

# 1. Load Data
from datasets import Dataset
from data import load_dataframe

# Firstly load dataframe

def clean_df():
    data = load_dataframe()

    # Drop missing values
    data = data.dropna(subset=['story', 'memType'])

    # load Dataset
    dataset = Dataset.from_pandas(data[['story', 'memType']])
    return dataset

# id2label = {0: "imagined", 1: "recalled", 2:"retold"}
# label2id = {"imagined":0, "recalled": 1, "retold":2}

# 2. Preprocessing
def tokenizer_function(dataset):
    """
    Args:
        dataset: Sequence of string to tokenize
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(dataset['story'], padding=True, truncation=True)

def tokenize_data(data):
    return data.map(tokenizer_function, batched = True)

def padding(tokenizer):
    """Returns data collator object
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)
