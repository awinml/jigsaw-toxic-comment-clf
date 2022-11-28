import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFXLMRobertaModel, XLMRobertaConfig
from transformers import AutoTokenizer, XLMRobertaTokenizer

from utils import generate_random_seed
from utils import regular_encode
from utils import load_train_set
from utils import load_test_set
from utils import build_dataset
from utils import build_classifier
from utils import show_training_process
from utils import train_classifier
from utils import predict_with_classifier
from utils import show_roc_auc

# Checking for TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    model_name = "jplu/tf-xlm-roberta-large"
    max_seq_len = 256
    batch_size_for_xlmr = 8 * strategy.num_replicas_in_sync
else:
    strategy = tf.distribute.get_strategy()
    physical_devices = tf.config.list_physical_devices("GPU")
    for device_idx in range(strategy.num_replicas_in_sync):
        tf.config.experimental.set_memory_growth(physical_devices[device_idx], True)
    max_seq_len = 256
    model_name = "jplu/tf-xlm-roberta-base"
    batch_size_for_xlmr = 4 * strategy.num_replicas_in_sync


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tfa.register_all()

xlmr_learning_rate = 1e-5
dataset_dir = "jigsaw-multilingual-toxic-comment-classification"
final_classifier_name = "xlmr_for_toxicity.h5"

xlmroberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
xlmroberta_config = XLMRobertaConfig.from_pretrained(model_name)

sentence_embedding_size = xlmroberta_config.hidden_size
assert max_seq_len <= xlmroberta_config.max_position_embeddings

corpus_for_training = load_train_set(
    os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"),
    text_field="comment_text",
    lang_field="lang",
    sentiment_fields=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
)
assert "en" in corpus_for_training

multilingual_corpus = load_train_set(
    os.path.join(dataset_dir, "validation.csv"),
    text_field="comment_text",
    lang_field="lang",
    sentiment_fields=[
        "toxic",
    ],
)
assert "en" not in multilingual_corpus
max_size = 0
print("Multilingual data:")
for language in sorted(list(multilingual_corpus.keys())):
    print("  {0}\t\t{1} samples".format(language, len(multilingual_corpus[language])))
    assert set(map(lambda cur: cur[1], multilingual_corpus[language])) == {0, 1}
    if len(multilingual_corpus[language]) > max_size:
        max_size = len(multilingual_corpus[language])


nonenglish_languages = sorted(list(multilingual_corpus.keys()))
corpus_for_validation = dict()
for lang in nonenglish_languages:
    random.shuffle(multilingual_corpus[lang])
    n = len(multilingual_corpus[lang]) // 2
    corpus_for_validation[lang] = multilingual_corpus[lang][0:n]
    corpus_for_training[lang] = multilingual_corpus[lang][n:]
    del multilingual_corpus[lang]

texts_for_submission = load_test_set(
    os.path.join(dataset_dir, "test.csv"),
    text_field="content",
    lang_field="lang",
    id_field="id",
)

for language in sorted(list(texts_for_submission.keys())):
    print("  {0}\t\t{1} samples".format(language, len(texts_for_submission[language])))

dataset_for_training, n_batches_per_data = build_dataset(
    texts=corpus_for_training,
    dataset_size=150000,
    tokenizer=xlmroberta_tokenizer,
    maxlen=max_seq_len,
    batch_size=batch_size_for_xlmr,
    shuffle=True,
)

dataset_for_validation, n_batches_per_epoch = build_dataset(
    texts=corpus_for_validation,
    dataset_size=6000,
    tokenizer=xlmroberta_tokenizer,
    maxlen=max_seq_len,
    batch_size=batch_size_for_xlmr,
    shuffle=False,
)


preparing_duration = int(round(time.time() - experiment_start_time))
print(
    "Duration of data loading and preparing to the Siamese NN training is "
    "{0} seconds.".format(preparing_duration)
)


with strategy.scope():
    xlmr_based_classifier = build_classifier(
        transformer_name=model_name,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=xlmr_learning_rate,
    )


train_classifier(
    nn=xlmr_based_classifier,
    trainset=dataset_for_training,
    steps_per_trainset=n_batches_per_data,
    steps_per_epoch=min(5 * n_batches_per_epoch, n_batches_per_data),
    validset=dataset_for_validation,
    max_duration=int(round(2.0 * 3600.0 - preparing_duration)),
    classifier_file_name=final_classifier_name,
)

val_predictions = predict_with_classifier(
    texts=corpus_for_validation,
    tokenizer=xlmroberta_tokenizer,
    maxlen=max_seq_len,
    classifier=xlmr_based_classifier,
    batch_size=batch_size_for_xlmr,
)


calculated_probas = []
true_labels = []
for lang in val_predictions:
    probabilities_, true_labels_ = val_predictions[lang]
    calculated_probas.append(probabilities_)
    true_labels.append(true_labels_)
calculated_probas = np.concatenate(calculated_probas)
true_labels = np.concatenate(true_labels)


show_roc_auc(y_true=true_labels, probabilities=calculated_probas, label="multi")

final_predictions = predict_with_classifier(
    texts=texts_for_submission,
    tokenizer=xlmroberta_tokenizer,
    maxlen=max_seq_len,
    classifier=xlmr_based_classifier,
    batch_size=batch_size_for_xlmr,
)
