# Data Exploration & Analysis

# Import libraries
import pandas as pd
import numpy as np

# load the data
data = pd.read_csv("./data/hcV3-stories.csv", sep=";")
data.head()

# Missing Values Treatment
data.isnull().sum()
print(data.isnull().sum())
data['annotatorAge'] = data['annotatorAge'].fillna(data['annotatorAge'].mean())
data['frequency'] = data['frequency'].fillna(data['frequency'].mean())
data['importance'] = data['importance'].fillna(data['importance'].mean())

# For our two important feartures let's drop the missing rows.
data = data.dropna(subset = ['story', 'memType'])
data[['story', 'memType']].isnull().sum()

# Preprocessing

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Load Spacy's English language model
# Download (large) English Pipeline from spacy
!python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# Create a list of the story per participant
docs = []

for doc in nlp.pipe(data['story'], n_process=os.cpu_count()-1, batch_size=200, disable = ["transformer", "ner", "textcat"]):
    docs.append(doc)


# Define a function to lemmatize content words
def preprocess_text(doc):
    # Lowercase and remove punctuation, double space, stop words and lemmatize text:
    clean_tokens = [token.lemma_.lower() for token in doc if not (token.is_punct or token.is_space or token.is_stop)]
    # Join the cleaned tokens back into a string
    clean_text = " ".join(clean_tokens)
    return clean_tokens, clean_text

# Apply the preprocessing pipeline using nlp.pipe
clean_tokens = []
clean_text = []

for doc in docs:
    tokens, text = preprocess_text(doc)
    clean_tokens.append(tokens)
    clean_text.append(text)

# Add the preprocessed text as a new column in the dataframe
data['story_preprocessed_text'] = clean_text
data['story_preprocessed_token'] = clean_tokens

all_tokens = [token for tokens in clean_tokens for token in tokens]
unique_tokens = set(all_tokens)

# COMPUTE THE NUMBER OF WORDS

# Number of words per story:
data['word_count'] = [len(text.split()) for text in data['story_preprocessed']]

# Compute the minimum, maximum, average and std number of word for all the stories
print('All stories in Hippocorpus')
print()
data['word_count'].describe()

# split real and imagined stories
data_recalled = data[data['memType']=='recalled']
data_imagined = data[data['memType']=='imagined']
data_retold = data[data['memType']=='retold']

# Whisker:  Min-Max number of words in Hippocorpus

word_count_description = data['word_count'].describe()

Q1 = np.percentile(data['word_count'], 25)    # word_count_description.loc['25%']
Q3 = np.percentile(data['word_count'], 75)    # word_count_description.loc['75%']
IQR = Q3 - Q1

min_thr = Q1 - 1.5*IQR
max_thr = Q3 + 1.5*IQR

# Let's check the stories that have number of words below 2.5 std
data[data['word_count'] < min_thr].loc[:,['memType','story','story_preprocessed','word_count']].sort_values('word_count', ascending=True, inplace=False)
# Let's check the stories that have a number of words below 2.5 std
data[data['word_count'] > max_thr].loc[:,['memType','story','story_preprocessed','word_count']].sort_values('word_count', ascending=False, inplace=False)

# Not include data without words after preprocessing
data = data[data['word_count'] != 0]



# Text transformation into vectors
