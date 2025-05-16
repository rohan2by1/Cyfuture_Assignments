# plagiarism_siamese.py

# --- Imports ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('punkt')

# --- Load and Clean Data ---

# Use tab separator, no header in file, assign names
DATA_PATH = "train/train.csv"
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text1', 'text2', 'label'])

print("---- First lines ----")
print(df.head())

# Drop rows with missing data in any column, especially label
df = df.dropna(subset=['text1', 'text2', 'label'])

# Make sure label is integer (0 or 1)
# If it's float due to CSV, convert to int, otherwise will throw error if nonconvertible values exist
try:
    df['label'] = df['label'].astype(int)
except Exception as e:
    print("Label conversion failed. Values in label column:", df['label'].unique())
    raise e

# --- Preprocessing (Tokenization & Padding) ---

max_num_words = 15000
max_seq_length = 250

all_texts = df['text1'].tolist() + df['text2'].tolist()
tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(all_texts)

def get_sequences(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_seq_length)

X1 = get_sequences(df['text1'])
X2 = get_sequences(df['text2'])
y = df['label'].values

# --- Train/Test Split ---

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Siamese LSTM Model ---

embedding_dim = 128

input_seq = Input(shape=(max_seq_length,))
x = Embedding(max_num_words, embedding_dim, mask_zero=True)(input_seq)
x = Bidirectional(LSTM(64))(x)
shared_model = Model(input_seq, x)

input_1 = Input(shape=(max_seq_length,))
input_2 = Input(shape=(max_seq_length,))
out_1 = shared_model(input_1)
out_2 = shared_model(input_2)

l1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([out_1, out_2])
out = Dense(64, activation='relu')(l1_distance)
out = Dropout(0.5)(out)
final_out = Dense(1, activation='sigmoid')(out)

model = Model([input_1, input_2], final_out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Training ---

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_test, X2_test], y_test),
    epochs=3,
    batch_size=64
)

# --- Evaluation ---

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

y_pred = (model.predict([X1_test, X2_test]) > 0.5).astype('int')
print(classification_report(y_test, y_pred))

# --- Working Demo ---

def demo(idx=0):
    prob = model.predict([X1_test[idx:idx+1], X2_test[idx:idx+1]])[0][0]
    true_lbl = y_test[idx]
    print(f"Predicted plagiarism probability: {prob:.2f} (True label: {true_lbl})")
    print("----TEXT 1----\n", df.iloc[X1_train.shape[0]+idx]['text1'][:300])
    print("----TEXT 2----\n", df.iloc[X1_train.shape[0]+idx]['text2'][:300])

# Try the first test pair
demo(0)