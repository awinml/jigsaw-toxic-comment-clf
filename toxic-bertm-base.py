import numpy as np
import pandas as pd
from utils import plot_learning_evolution
import transformers
import tensorflow as tf

train = pd.read_csv(
    "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"
)

validation = pd.read_csv(
    "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv"
)

X_train = train["comment_text"]
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_test = validation["comment_text"]
y_test = validation[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]


def map_func(input_ids, masks, labels):
    return {"input_ids": input_ids, "attention_mask": masks}, labels


model_checkpoint = "bert-base-multilingual-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

seq_len = 128
X_train_ids = np.zeros((len(X_train), seq_len))
X_train_mask = np.zeros((len(X_train), seq_len))

X_test_ids = np.zeros((len(X_test), seq_len))
X_test_mask = np.zeros((len(X_test), seq_len))

for index, sequence in enumerate(X_train["comment_text"]):
    tokens = tokenizer.encode_plus(
        sequence,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="tf",
    )
    X_train_ids[index, :], X_train_mask[index, :] = (
        tokens["input_ids"],
        tokens["attention_mask"],
    )

for index, sequence in enumerate(X_test["comment_text"]):
    tokens = tokenizer.encode_plus(
        sequence,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="tf",
    )
    X_test_ids[index, :], X_test_mask[index, :] = (
        tokens["input_ids"],
        tokens["attention_mask"],
    )

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_ids, X_train_mask, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_ids, X_test_mask, y_test))

train_dataset = train_dataset.map(map_func)
test_dataset = test_dataset.map(map_func)

bert = transformers.TFAutoModel.from_pretrained(model_checkpoint)

input_ids = tf.keras.layers.Input(shape=(seq_len,), name="input_ids", dtype="int32")
mask = tf.keras.layers.Input(shape=(seq_len,), name="attention_mask", dtype="int32")

embeddings = bert.bert(input_ids, attention_mask=mask)[1]

X = tf.keras.layers.Dense(1024, activation="relu")(embeddings)
y = tf.keras.layers.Dense(6, activation="sigmoid", name="outputs")(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

model.layers[2].trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-5, decay=1e-6),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="AUC")],
)

history = model.fit(
    train_dataset, validation_data=(test_dataset), epochs=10, batch_size=64
)

model.evaluate(test_dataset)

plot_learning_evolution(history)
