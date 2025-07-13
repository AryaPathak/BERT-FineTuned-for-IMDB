

# 🎬 BERT Fine-Tuning on IMDB Sentiment Dataset (TensorFlow)

This project demonstrates how to fine-tune a pre-trained **BERT model** (`bert-base-uncased`) on the **IMDB movie review dataset** using **TensorFlow** and **Hugging Face Transformers**. The goal is to classify reviews as **positive** or **negative**.

---

## 📁 Project Structure

```

.
├── app.py                   # Main training script
├── bert-imdb-finetuned/    # Directory where the fine-tuned model is saved
├── requirements.txt         # Python dependencies
└── README.md                # This file

````

---

## 🚀 Features

- Loads BERT from Hugging Face Model Hub
- Downloads IMDB dataset automatically
- Preprocesses text using BERT tokenizer
- Converts Hugging Face dataset to TensorFlow `tf.data.Dataset`
- Fine-tunes using `TFAutoModelForSequenceClassification`
- Saves the fine-tuned model and tokenizer for later use

---

## 🧠 Model Details

- **Model**: `bert-base-uncased`
- **Task**: Binary Sentiment Classification
- **Dataset**: [IMDB](https://huggingface.co/datasets/imdb)
- **Frameworks**: TensorFlow 2.x, Hugging Face Transformers

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/bert-imdb-finetuning.git
cd bert-imdb-finetuning
python -m venv .venv
source .venv/Scripts/activate  # or `source .venv/bin/activate` on Unix
pip install -r requirements.txt
````

If `requirements.txt` is missing, install directly:

```bash
pip install transformers datasets tensorflow tf-keras
```

---

## ▶️ How to Run

```bash
python app.py
```

This will:

* Load the pretrained model and tokenizer
* Download and preprocess the IMDB dataset
* Train the model for 3 epochs
* Save the fine-tuned model to `bert-imdb-finetuned/`

---

## 💾 Output

After training, the following files will be saved:

```
bert-imdb-finetuned/
├── config.json
├── tf_model.h5
├── tokenizer_config.json
├── vocab.txt
```

You can reload the model using:

```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("bert-imdb-finetuned")
tokenizer = AutoTokenizer.from_pretrained("bert-imdb-finetuned")
```

---

## 📊 Sample Performance

| Epoch | Accuracy (val) |
| ----- | -------------- |
| 1     | \~0.87         |
| 2     | \~0.90         |
| 3     | \~0.91         |

---

## 📌 TODO

* [ ] Add inference script for new reviews
* [ ] Create a simple Gradio UI for demo
* [ ] Push model to Hugging Face Hub

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

* 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers)
* 📚 [IMDB Dataset](https://huggingface.co/datasets/imdb)
* 🧠 [TensorFlow](https://www.tensorflow.org/)

