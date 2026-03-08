# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install sentence_transformers

# importing all required libraries
import tensorflow as tf
import keras
import time 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.decomposition import PCA
import pickle as pk

#Initialize the objects here
model = SentenceTransformer('bert-base-nli-mean-tokens') #BERT Sentence Transformer
tokenizer = T5Tokenizer.from_pretrained("t5-large") #T5 Pretrained Tokenizer
model_2 = T5ForConditionalGeneration.from_pretrained("t5-large") #T5 Conditional Generated


# Reading the data from the files.
df = pd.read_csv('/kaggle/input/q-a-conversation/q_a_pairs.csv') #Question Answer pairs
q_pca = np.loadtxt('/kaggle/input/q-a-conversation/q_pca_embed.txt') #Vector embeddings of questions 
pca = pk.load(open("/kaggle/input/q-a-conversation/pca.pkl",'rb')) # pca model for reducing dimensions of user input

# dataframe datatype fixed, and irrelevant Index columns removed
df['Answer'] = df['Answer'].astype(str)
df = df[['Question', 'Answer']]
