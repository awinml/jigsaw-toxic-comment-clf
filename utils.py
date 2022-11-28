import codecs
import copy
import csv
import gc
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFXLMRobertaModel, XLMRobertaConfig
from transformers import AutoTokenizer, XLMRobertaTokenizer


def generate_random_seed() -> int:
    """
    generate_random_seed

    Returns
    -------
    int
        Random integer
    """
    return random.randint(0, 2147483648)


def regular_encode(
    texts: List[str], tokenizer: XLMRobertaTokenizer, maxlen: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    regular_encode Tokenize inputs for XLMRoberta Model

    Parameters
    ----------
    texts : List[str]
        Texts to be tokenized.
    tokenizer : XLMRobertaTokenizer
        Model Tokenizer Instance.
    maxlen : int
        Max token lenght.

    Returns
    -------
    encoded_tokens [np.ndarray]
        The encoded tokens.

    encoded_masks [np.ndarray]
        The masks for the encoded tokens.
    """
    err_msg = '"{0}" is wrong type for the text list!'.format(type(texts))
    assert isinstance(texts, list) or isinstance(texts, tuple), err_msg
    enc_di = tokenizer.batch_encode_plus(
        texts, return_token_type_ids=False, padding="max_length", max_length=maxlen
    )
    err_msg = "{0} != {1}".format(len(texts), len(enc_di["input_ids"]))
    assert len(texts) == len(enc_di["input_ids"]), err_msg
    err_msg = "{0} != {1}".format(len(texts), len(enc_di["attention_mask"]))
    assert len(texts) == len(enc_di["attention_mask"]), err_msg
    encoded_tokens = np.zeros((len(texts), maxlen), dtype=np.int32)
    encoded_masks = np.zeros((len(texts), maxlen), dtype=np.int32)
    for sample_idx, (encoded_cur_text, encoded_cur_mask) in enumerate(
        zip(enc_di["input_ids"], enc_di["attention_mask"])
    ):
        n_text = len(encoded_cur_text)
        n_mask = len(encoded_cur_mask)
        err_msg = 'Tokens and masks of texts "{0}" are different! ' "{1} != {2}".format(
            texts[sample_idx], n_text, n_mask
        )
        assert n_text == n_mask, err_msg
        if n_text >= maxlen:
            encoded_tokens[sample_idx] = np.array(
                encoded_cur_text[0:maxlen], dtype=np.int32
            )
            encoded_masks[sample_idx] = np.array(
                encoded_cur_mask[0:maxlen], dtype=np.int32
            )
        else:
            padding = [0 for _ in range(maxlen - n_text)]
            encoded_tokens[sample_idx] = np.array(
                encoded_cur_text + padding, dtype=np.int32
            )
            encoded_masks[sample_idx] = np.array(
                encoded_cur_mask + padding, dtype=np.int32
            )
    return encoded_tokens, encoded_masks


def load_train_set(
    file_name: str, text_field: str, sentiment_fields: List[str], lang_field: str
) -> Dict[str, List[Tuple[str, int]]]:
    assert len(sentiment_fields) > 0, "List of sentiment fields is empty!"
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=",")
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    for cur_field in sentiment_fields:
                        err_msg2 = err_msg + ' Field "{0}" is not found!'.format(
                            cur_field
                        )
                        assert cur_field in header, err_msg2
                    text_field_index = header.index(text_field)
                    try:
                        lang_field_index = header.index(lang_field)
                    except:
                        lang_field_index = -1
                    indices_of_sentiment_fields = []
                    for cur_field in sentiment_fields:
                        indices_of_sentiment_fields.append(header.index(cur_field))
                else:
                    if len(row) == len(header):
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + " Text is empty!"
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + " Language is empty!"
                        else:
                            cur_lang = "en"
                        max_proba = 0.0
                        for cur_field_idx in indices_of_sentiment_fields:
                            try:
                                cur_proba = float(row[cur_field_idx])
                            except:
                                cur_proba = -1.0
                            err_msg2 = err_msg + " Value {0} is wrong!".format(
                                row[cur_field_idx]
                            )
                            assert (cur_proba >= 0.0) and (cur_proba <= 1.0), err_msg2
                            if cur_proba > max_proba:
                                max_proba = cur_proba
                        new_label = 1 if max_proba >= 0.5 else 0
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, new_label))
            if line_idx % 10000 == 0:
                print(
                    '{0} lines of the "{1}" have been processed...'.format(
                        line_idx, file_name
                    )
                )
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print(
                '{0} lines of the "{1}" have been processed...'.format(
                    line_idx - 1, file_name
                )
            )
    return data_by_lang


def load_test_set(
    file_name: str, id_field: str, text_field: str, lang_field: str
) -> Dict[str, List[Tuple[str, int]]]:
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=",")
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(id_field)
                    assert id_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(lang_field)
                    assert lang_field in header, err_msg2
                    id_field_index = header.index(id_field)
                    text_field_index = header.index(text_field)
                    lang_field_index = header.index(lang_field)
                else:
                    if len(row) == len(header):
                        try:
                            id_value = int(row[id_field_index])
                        except:
                            id_value = -1
                        err_msg2 = err_msg + " {0} is wrong ID!".format(
                            row[id_field_index]
                        )
                        assert id_value >= 0, err_msg2
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + " Text is empty!"
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + " Language is empty!"
                        else:
                            cur_lang = "en"
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, id_value))
            if line_idx % 10000 == 0:
                print(
                    '{0} lines of the "{1}" have been processed...'.format(
                        line_idx, file_name
                    )
                )
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print(
                '{0} lines of the "{1}" have been processed...'.format(
                    line_idx - 1, file_name
                )
            )
    return data_by_lang


def build_dataset(
    texts: Dict[str, List[Tuple[str, int]]],
    dataset_size: int,
    tokenizer: XLMRobertaTokenizer,
    maxlen: int,
    batch_size: int,
    shuffle: bool,
) -> Tuple[tf.data.Dataset, int]:
    texts_and_labels = []
    dataset_size_by_lang = int(round(dataset_size / float(len(texts))))
    for lang in texts:
        print("{0}:".format(lang))
        n_lang = 0
        if shuffle:
            if len(texts[lang]) > dataset_size_by_lang:
                texts_and_labels += random.sample(texts[lang], k=dataset_size_by_lang)
                n_lang += dataset_size_by_lang
            elif len(texts[lang]) < dataset_size_by_lang:
                texts_and_labels += texts[lang]
                n_lang += len(texts[lang])
                n = dataset_size_by_lang - len(texts[lang])
                while n >= len(texts[lang]):
                    texts_and_labels += texts[lang]
                    n -= len(texts[lang])
                    n_lang += len(texts[lang])
                if n > 0:
                    texts_and_labels += random.sample(texts[lang], k=n)
                    n_lang += n
            else:
                texts_and_labels += texts[lang]
                n_lang += len(texts[lang])
        else:
            texts_and_labels += texts[lang]
            n_lang += len(texts[lang])
        print("  number of samples is {0};".format(n_lang))
    random.shuffle(texts_and_labels)
    n_steps = len(texts_and_labels) // batch_size
    print("Samples number of the data set is {0}.".format(len(texts_and_labels)))
    tokens_of_texts, mask_of_texts = regular_encode(
        texts=[cur[0] for cur in texts_and_labels], tokenizer=tokenizer, maxlen=maxlen
    )
    toxicity_labels = np.array([cur[1] for cur in texts_and_labels], dtype=np.int32)
    print(
        "Number of positive siamese samples is {0} from {1}.".format(
            int(sum(toxicity_labels)), toxicity_labels.shape[0]
        )
    )
    if shuffle:
        err_msg = "{0} is too small number of samples for the data set!".format(
            len(texts_and_labels)
        )
        assert n_steps >= 50, err_msg
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                ((tokens_of_texts, mask_of_texts), toxicity_labels)
            )
            .repeat()
            .batch(batch_size)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            ((tokens_of_texts, mask_of_texts), toxicity_labels)
        ).batch(batch_size)
    del texts_and_labels
    return dataset, n_steps


def build_classifier(transformer_name: str, max_len: int, lr: float) -> tf.keras.Model:
    """
    build_classifier Create Keras model.

    Parameters
    ----------
    transformer_name : str
        Transformer model to be used.
    max_len : int
        Max token length for transformer.
    lr : float
        Rearning rate.

    Returns
    -------
    tf.keras.Model
        Keras model instance.
    """
    word_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="base_word_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="base_attention_mask"
    )
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name, name="Transformer"
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    pooled_output = tf.keras.layers.GlobalAvgPool1D(name="AvePool")(
        sequence_output, mask=attention_mask
    )
    kernel_init = tf.keras.initializers.GlorotNormal(seed=generate_random_seed())
    bias_init = tf.keras.initializers.Constant(value=0.0)
    cls_layer = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name="OutputLayer",
    )(pooled_output)
    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask], outputs=cls_layer, name="ToxicityClassifier"
    )
    cls_model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss="binary_crossentropy",
    )
    return cls_model


def show_training_process(
    history: tf.keras.callbacks.History, metric_name: str, figure_id: int = 1
):
    """
    show_training_process
        Compute metrics and plot validation curves

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Keras model history object.
    metric_name : str
        Name of the metric to be computed.
    figure_id : int, optional
        Unique identifier for plot, by default 1.
    """
    val_metric_name = "val_" + metric_name
    err_msg = 'The metric "{0}" is not found! Available metrics are: {1}'.format(
        metric_name, list(history.history.keys())
    )
    assert metric_name in history.history, err_msg
    plt.figure(figure_id, figsize=(5, 5))
    plt.plot(
        list(range(len(history.history[metric_name]))),
        history.history[metric_name],
        label="Training {0}".format(metric_name),
    )
    if val_metric_name in history.history:
        assert len(history.history[metric_name]) == len(
            history.history["val_" + metric_name]
        )
        plt.plot(
            list(range(len(history.history["val_" + metric_name]))),
            history.history["val_" + metric_name],
            label="Validation {0}".format(metric_name),
        )
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title("Training process")
    plt.legend(loc="best")
    plt.show()


def train_classifier(
    nn: tf.keras.Model,
    trainset: tf.data.Dataset,
    steps_per_trainset: int,
    steps_per_epoch: int,
    validset: tf.data.Dataset,
    max_duration: int,
    classifier_file_name: str,
):
    """
    train_classifier
        Train Keras model

    Parameters
    ----------
    nn : tf.keras.Model
        Keras model instance
    trainset : tf.data.Dataset
        Train data
    steps_per_trainset : int
    steps_per_epoch : int
        Steps per epoch.
    validset : tf.data.Dataset
        validation Data
    max_duration : int
        Estimated time of training.
    classifier_file_name : str
        Model name
    """
    assert steps_per_trainset >= steps_per_epoch
    n_epochs = int(round(10.0 * steps_per_trainset / float(steps_per_epoch)))
    print(
        f"Maximal duration of the XLMR-based classifier training is {max_duration} seconds."
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=7,
            monitor="val_loss",
            mode="min",
            restore_best_weights=False,
            verbose=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            save_best_only=True,
            filepath=classifier_file_name,
        ),
        tfa.callbacks.TimeStopping(seconds=max_duration, verbose=True),
    ]
    history = nn.fit(
        trainset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validset,
        epochs=n_epochs,
        callbacks=callbacks,
    )
    show_training_process(history, "loss")
    nn.load_weights(classifier_file_name)


def predict_with_classifier(
    texts: Dict[str, List[Tuple[str, int]]],
    tokenizer: XLMRobertaTokenizer,
    maxlen: int,
    classifier: tf.keras.Model,
    batch_size: int,
    max_dataset_size: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    languages = sorted(list(texts.keys()))
    predictions_by_languages = dict()
    if max_dataset_size > 0:
        max_size_per_lang = max_dataset_size // len(languages)
        err_msg = "{0} is too small number of dataset samples!".format(max_dataset_size)
        assert max_size_per_lang > 0, err_msg
    else:
        max_size_per_lang = 0
    for cur_lang in languages:
        selected_indices = list(range(len(texts[cur_lang])))
        if max_size_per_lang > 0:
            if len(selected_indices) > max_size_per_lang:
                selected_indices = random.sample(
                    population=selected_indices, k=max_size_per_lang
                )
        tokens_of_texts, mask_of_texts = regular_encode(
            texts=[texts[cur_lang][idx][0] for idx in selected_indices],
            tokenizer=tokenizer,
            maxlen=maxlen,
        )
        predictions = []
        n_batches = int(np.ceil(len(selected_indices) / float(batch_size)))
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(len(selected_indices), batch_start + batch_size)
            res = classifier.predict_on_batch(
                [
                    tokens_of_texts[batch_start:batch_end],
                    mask_of_texts[batch_start:batch_end],
                ]
            )
            if not isinstance(res, np.ndarray):
                res = res.numpy()
            predictions.append(res.reshape((res.shape[0],)))
            del res
        predictions = np.concatenate(predictions)
        identifiers = np.array(
            [texts[cur_lang][idx][1] for idx in selected_indices], dtype=np.int32
        )
        predictions_by_languages[cur_lang] = (predictions, identifiers)
        del predictions, identifiers, selected_indices
    return predictions_by_languages


def show_roc_auc(
    y_true: np.ndarray, probabilities: np.ndarray, label: str, figure_id: int = 1
):
    plt.figure(figure_id, figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--")
    print(
        "ROC-AUC score for {0} is {1:.9f}".format(
            label, roc_auc_score(y_true=y_true, y_score=probabilities)
        )
    )
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=probabilities)
    plt.plot(fpr, tpr, label=label.title())
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.show()


def plot_learning_evolution(r):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(r.history["loss"], label="Loss")
    plt.plot(r.history["val_loss"], label="val_Loss")
    plt.title("Loss evolution during trainig")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(r.history["AUC"], label="AUC")
    plt.plot(r.history["val_AUC"], label="val_AUC")
    plt.title("AUC score evolution during trainig")
    plt.legend()
