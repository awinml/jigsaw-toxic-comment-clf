import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train = pd.read_csv(
    "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"
)
train.drop(
    ["severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    axis=1,
    inplace=True,
)
validation = pd.read_csv(
    "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv"
)

X_train = train["comment_text"]
y_train = train["toxic"]

X_test = validation["comment_text"]
y_test = validation["toxic"]

# Tokenize text

max_features = 500
tokenizer = Tokenizer(num_words=max_features)
maxlen = 512

tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)

# Padding the sequences
X_train_pad = pad_sequences(X_train_tokenized, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_tokenized, maxlen=maxlen)

embed_size = 64

# A simpleRNN without any pretrained embeddings and one dense layer
model = Sequential()
model.add(Embedding(max_features, embed_size, input_length=maxlen))
model.add(SimpleRNN(2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


history = model.fit(
    X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test)
)

y_pred = model.predict(X_test_pad)

threshold = 0.5
y_pred_encoded = np.where(y_pred >= threshold, 1, 0)
auc = round(roc_auc_score(y_pred_encoded, y_test), 2)
print(f"ROC AUC Score: {auc}")
