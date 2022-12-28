# Jigsaw - Multilingual Toxic Comment Classification

- Built a **multilingual text classification model** to predict the probability that a comment is toxic using the data provided by Google Jigsaw.

- The data had **4,35,775** text comments in **7 different languages**.

- A RNN model was used as a baseline. The **BERT-Multilingual-base and XLMRoBERTa** models were fine-tuned to get the best results.

- The best results were obtained using the fine-tuned XLMRoberta model. It achieved an **Accuracy of 96.24% and an ROC-AUC Score of 93.92%.**

## Data

It only takes one toxic comment to sour an online discussion. Toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could have a safer, more collaborative internet.

The goal is to find the probability that a comment is toxic.

#### Columns in the dataset:

    id - identifier within each file.
    comment_text - the text of the comment to be classified.
    lang - the language of the comment.
    toxic - whether or not the comment is classified as toxic.

The comments are composed of multiple non-English languages and come either from Civil Comments or Wikipedia talk page edits.

The dataset can be downloaded from [here](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification).

## Experiments:

#### **RNN:**

- A baseline was created using the RNN model. An **embedding layer of size 64** was used. Training the model with an **Adam optimizer with learning rate of 0.001** for **5 epochs** yielded an **Accuracy of 83.68% and an ROC-AUC Score of 55.72%.**

#### **BERT-Multilingual-base:**

- The BERT-Multilingual-base was fine tuned on the data. A **hidden layer of 1024 neurons** was added to the model. Training the model with an **Adam optimizer with learning rate of 0.001, weight decay of 1e-6** for **10 epochs** yielded an **Accuracy of 93.92% and an ROC-AUC Score of 89.55%.**

#### **XLM RoBERTa:**

- The XLMRoberta model was fine tuned on the data. An **embedding layer of size 768** was used. Training the model with an **AdamW optimizer with learning rate of 1e-5, weight decay of 1e-5** for **7 epochs** yielded an **Accuracy of 96.24% and an ROC-AUC Score of 93.92%.**

For all the models that were fine-tuned:

- Maximum input sequence length was 512.
- Batch size of 64 was used for training.
- Binary Cross-Entropy was used as the loss function.

## Results:

The best results were obtained using a fine-tuned XLMRoberta model. It was used for generating the final predictions. It achieved an **Accuracy of 96.24% and an ROC-AUC Score of 93.92%.**

The results from all the models have been summarized below:

|                  **Model**                  | **Accuracy** | **ROC\-AUC Score** |
| :-----------------------------------------: | :----------: | :----------------: |
|                   **RNN**                   |    83.68     |       88.72        |
| **BERT-Multilingual-base** _\(fine-tuned\)_ |    93.92     |       89.55        |
|      **XLM RoBERTa** _\(fine-tuned\)_       |  **96.24**   |     **93.92**      |

## Run Locally

1. Install required libraries:
   ```bash
     pip install -r requirements.txt
   ```
2. Baseline model:
   ```bash
     python toxic-baseline-rnn.py
   ```
3. Fine-tune models:
   ```bash
     python toxic-bertm-base.py
     python toxic-xlm-roberta.py
   ```

## License &nbsp;&nbsp; [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Author: [@awinml](https://www.github.com/awinml)

## Feedback

If you have any feedback, please reach out to me at: &nbsp; &nbsp;
<a href="https://www.linkedin.com/in/ashwin-mathur-ds/"><img src="https://img.shields.io/badge/LinkedIn-Ashwin-blue" alt="LinkedIn" href="https://www.linkedin.com/in/ashwin-mathur-ds/"></a>
