from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from data import import_data
from config import PATH_TO_DATA

def clean_df():
    data = import_data(PATH_TO_DATA)

    # Drop missing values
    data = data.dropna(subset=['story', 'memType'])
    print(data)
    # load Dataset
    dataset = Dataset.from_pandas(data[['story', 'memType']])
    return dataset

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

if __name__ == '__main__':
    clean_dataset = clean_df()
    tokenized = tokenize_data(clean_dataset)
    padded_data = padding(tokenized)
    print(padded_data)
