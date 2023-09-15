# installing packages
# !pip install transforms datasets
# !pip install evaluate
# !pip install accelerate -U
# !pip install transformers[torch]

# 1. Load Data
from datasets import Dataset
from data import load_dataframe

# Firstly load dataframe
data = load_dataframe()

#dataset2= load_my_dataset()

# load Dataset
dataset = Dataset.from_pandas(data[['story', 'memType']])
dataset = dataset.rename_column('memType', 'labels')
dataset = dataset.rename_column('story', 'text')

# id2label = {0: "imagined", 1: "recalled", 2:"retold"}
# label2id = {"imagined":0, "recalled": 1, "retold":2}

# 2. Preprocessing
from transformers import AutoTokenizer, DistilBertTokenizerFast

# create an instance of tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased") 

# define a function to tokinize the dataset
def tokensizer_function(dataset):
    """
    Args:
        dataset: Sequence of string to tokenize
    """
    return tokenizer(dataset['text'], padding='max_length', truncation=True)

# Apply the function to the datasets
tokenized_dataset = dataset.map(tokensizer_function, batched = True)


