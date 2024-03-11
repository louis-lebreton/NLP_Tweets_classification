"""
NLP Tweets classification

Classify tweets between those related to natural disasters and those not related to them using NLP

Packages
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimpy import skim

from sklearn.model_selection import train_test_split # creation train / test data
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFBertModel
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from keras.losses import BinaryCrossentropy


# Repository
os.chdir('C:/Users/lebre/OneDrive/Bureau/Portfolio/Projets/NLP_disaster_tweets_classification')

# Data importation
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')

##############################################################################################################
## EDA #######################################################################################################
##############################################################################################################

df_train.shape
df_train.head
df_train.memory_usage().sum() / 1024**2

df_test.shape # no target
df_test.memory_usage().sum() / 1024**2

df_train.head
df_train['text'].head # tweet
df_train['target'].head # target : natural disaster or not

# missing values
print(df_train.isna().sum())

# disaster proportion plot
plt.figure(figsize =(8,6))
ax = plt.axes()
ax = sns.countplot(x ='target', data = df_train, palette = ['#4DC27D','#870D0D'])
plt.title('Disasters proportion',fontsize = 25)
plt.xlabel('0: No disaster 1: Disaster')
plt.ylabel('Count')
bbox_args = dict(boxstyle = 'round', fc = '0.9')

for p in ax.patches:
        ax.annotate('{:.0f}: {:.2f}%'.format(p.get_height(), (p.get_height() / len(df_train['target'])) * 100), (p.get_x() + 0.25, p.get_height() + 60), 
                   color = 'black',
                   bbox = bbox_args,
                   fontsize = 15)
plt.show()

# top 10 of most frequent location
top_10_loc = df_train['location'].value_counts()[:10]

plt.figure(figsize=(10, 6))
top_10_loc.plot(kind='bar', color='#0D6D87')
plt.title('Top 10 most represented cities')
plt.xlabel('Location')
plt.ylabel('Number of occurrences')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# top 10 of most frequent location among disasters
top_10_loc_disaster = df_train[df_train['target'] == 1]['location'].value_counts()[:10]

plt.figure(figsize=(10, 6))
top_10_loc_disaster.plot(kind='bar', color='#870D0D')
plt.title('Top 10 most represented cities among the disasters')
plt.xlabel('Location')
plt.ylabel('Number of occurrences')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

len_text_disaster_0=(df_train[df_train['target'] == 0])['text'].str.len()
len_text_disaster_1=(df_train[df_train['target'] == 1])['text'].str.len()

# describes of no disaster and disaster text length 
len_text_disaster_0.describe()
len_text_disaster_1.describe()

# distribution of no disaster and disaster text length 
plt.hist(len_text_disaster_0, bins=30,color='#4DC27D')
plt.title('Text length distribution - Disaster = 0')
plt.xlabel('text length')
plt.ylabel('frequency')
plt.show()

plt.hist(len_text_disaster_1, bins=30,color='#870D0D')
plt.title('Text length distribution - Disaster = 1')
plt.xlabel('text length')
plt.ylabel('frequency')
plt.show()

# 140 characters in pic because that was the maximum limit on the length of a tweet

##############################################################################################################
## NLP Model #################################################################################################
##############################################################################################################

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased') # pre-trained tokenizer for the BERT model
bert = TFBertModel.from_pretrained('bert-large-uncased') # for use with TF

# conversion of data into tokens
 
X_train = tokenizer(
    text = df_train['text'].tolist(),
    add_special_tokens = True,
    max_length = 36, # maximum length of tokenized sequences at 36. If a sequence is longer, it will be truncated
    truncation = True,
    padding = True, 
    return_tensors = 'tf',
    return_attention_mask = True,
    verbose = True)

X_test = tokenizer(
    text = df_test['text'].tolist(),
    add_special_tokens = True,
    max_length = 36, # maximum length of tokenized sequences at 36. If a sequence is longer, it will be truncated
    truncation = True,
    padding = True, 
    return_tensors = 'tf',
    return_attention_mask = True,
    verbose = True)


print(X_train)
print(X_test)
# we see the paddings (0) and the special tokens (101,102,...)
# we see the attention mask secondly
    
# creation of the model
input_ids = Input(shape=(36,),dtype=tf.int32,name='input_ids') # input for id
input_attention_mask = Input(shape=(36,),dtype=tf.int32,name='input_attention_mask') # input for attention mask


embeddings = bert(input_ids = input_ids, attention_mask = input_attention_mask)[0]

layer = layers.Dropout(0.2)(embeddings) # adds a Dropout layer to reduce overfitting, turning off 20% of neurons randomly during training
layer = layers.Dense(1024,activation = 'relu')(layer) # 1024 neurons
layer = layers.Dense(32,activation = 'relu')(layer) # 32 neurons
layer = layers.Flatten()(layer)
y = layers.Dense(1, activation = 'sigmoid')(layer) # last layer : 1 neuron : sigmoid for binary classification

# keras model
model = keras.Model(inputs = [input_ids, input_attention_mask], outputs = y)

model.summary()

# model compilation
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss = BinaryCrossentropy(from_logits=True), 
    metrics = ['accuracy']
)

# model training
classifier = model.fit(
    x = {'input_ids': X_train['input_ids'],
         'input_attention_mask': X_train['attention_mask']
        },
    y = df_train['target'].values,
    validation_split = 0.05, # 5% of training data for validation
    epochs = 5, # model goes through the training data set 5 times
    batch_size = 32 # data by batches of 32 examples at a times
)

# prediction
y_pred = model.predict({
    'input_ids': X_test['input_ids'],
    'input_attention_mask': X_test['attention_mask']
})

df_test['target']=(y_pred > 0.5).astype("int32")
# data exportation
df_test.to_csv('forecasts.csv',index=False)
