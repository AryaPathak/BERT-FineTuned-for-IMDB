from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import tensorflow as tf



# 1. Load Pretrained Model and Tokenizer
model_name = "bert-base-uncased"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load IMDB dataset from Hugging Face
dataset = load_dataset("imdb")

# 3. Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Convert to TensorFlow datasets
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=16
)

val_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=16
)

# 5. Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 6. Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# 7. Save the model
model.save_pretrained("bert-imdb-finetuned")
tokenizer.save_pretrained("bert-imdb-finetuned")