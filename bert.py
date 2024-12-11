# %% [markdown]
# # Dependencies

# %% [markdown]
# ## Modules

# %%
!pip install --no-cache-dir torch transformers datasets

# %%
!pip install --no-cache-dir seqeval==1.2.2 --use-pep517

# %%
!pip install --no-cache-dir optuna

# %%
!pip install --no-cache-dir torchviz


# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from seqeval.metrics import classification_report
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# %% [markdown]
# # Dataset

# %%
# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenization and alignment
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# %%
# DataLoader collate function
def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }

train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=16, collate_fn=collate_fn)

# %% [markdown]
# # Model

# %% [markdown]
# ## Training

# %%
class BertForNERWithLayerAttention(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BertForNERWithLayerAttention, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.num_hidden_layers = self.bert.config.num_hidden_layers + 1 
        self.layer_weights = nn.Parameter(torch.ones(self.num_hidden_layers))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, batch_size, seq_len, hidden_size)
        weighted_hidden_states = torch.sum(self.layer_weights[:, None, None, None] * hidden_states, dim=0)
        sequence_output = self.dropout(weighted_hidden_states)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)

        return {"loss": loss, "logits": logits}

# %%
# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForNERWithLayerAttention(pretrained_model_name="bert-base-cased", num_labels=num_labels).to(device)
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# %%
# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

# %% [markdown]
# ## Evaluation

# %%
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            predictions.append([label_list[p] for p, l in zip(pred, label) if l != -100])
            true_labels.append([label_list[l] for p, l in zip(pred, label) if l != -100])

print(classification_report(true_labels, predictions))

# %% [markdown]
# ## Hyperparameter Tuning

# %% [markdown]
# ### Optuna Study

# %%
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = 3

    # Define model with trial's hyperparameters
    class BertForNER(nn.Module):
        def __init__(self, pretrained_model_name, num_labels):
            super(BertForNER, self).__init__()
            self.bert = AutoModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
            self.num_hidden_layers = self.bert.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(self.num_hidden_layers))
            self.dropout = nn.Dropout(dropout_rate)  # Use trial's dropout rate
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, batch_size, seq_len, hidden_size)
            weighted_hidden_states = torch.sum(self.layer_weights[:, None, None, None] * hidden_states, dim=0)
            sequence_output = self.dropout(weighted_hidden_states)
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.size(-1))[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)

            return {"loss": loss, "logits": logits}

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForNER(pretrained_model_name="bert-base-cased", num_labels=num_labels).to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Prepare data loaders
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=collate_fn)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                predictions.append([label_list[p] for p, l in zip(pred, label) if l != -100])
                true_labels.append([label_list[l] for p, l in zip(pred, label) if l != -100])

    # Compute F1-score
    f1 = f1_score([label for seq in true_labels for label in seq],
                  [label for seq in predictions for label in seq],
                  average="weighted")
    return f1

# %%
# Create Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Try 20 different combinations

# Print best parameters
print("Best hyperparameters:", study.best_params)

# %% [markdown]
# ## Best Model

# %%
learning_rate = 4.369484270668493e-05
dropout_rate = 0.33027803609683487
batch_size = 32
num_epochs = 50

# %%
# Define the final model
class BertForFinalNER(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_rate):
        super(BertForFinalNER, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.num_hidden_layers = self.bert.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(self.num_hidden_layers))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        weighted_hidden_states = torch.sum(self.layer_weights[:, None, None, None] * hidden_states, dim=0)
        sequence_output = self.dropout(weighted_hidden_states)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)

        return {"loss": loss, "logits": logits}

# %% [markdown]
# ### Training

# %%
# Instantiate the model with the best dropout rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForFinalNER(pretrained_model_name="bert-base-cased", num_labels=num_labels, dropout_rate=dropout_rate).to(device)

# Define the optimizer with the best learning rate
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Define the learning rate scheduler
num_training_steps = num_epochs * len(train_loader)  # Total number of steps (batches * epochs)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Prepare data loaders with the best batch size
train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=collate_fn)

# Track loss values
train_losses = []
val_losses = []

# Early stopping parameters
patience = 10  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to qualify as improvement
best_val_loss = float("inf")
epochs_without_improvement = 0

# Training and validation loops
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # Update learning rate
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_val_loss += loss.item()

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                predictions.append([label_list[p] for p, l in zip(pred, label) if l != -100])
                true_labels.append([label_list[l] for p, l in zip(pred, label) if l != -100])

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

    # Check for improvement
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        print(f"Validation loss improved to {avg_val_loss:.4f}.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epochs.")

    # Early stopping
    if epochs_without_improvement >= patience:
        print("Early stopping triggered. Training stopped.")
        break

# Compute evaluation metrics
flattened_preds = [label for seq in predictions for label in seq]
flattened_labels = [label for seq in true_labels for label in seq]

accuracy = accuracy_score(flattened_labels, flattened_preds)
f1 = f1_score(flattened_labels, flattened_preds, average="weighted")

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid()
plt.show()

# %%
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Compute evaluation metrics
flattened_preds = [label for seq in predictions for label in seq]
flattened_labels = [label for seq in true_labels for label in seq]

# Compute accuracy
accuracy = accuracy_score(flattened_labels, flattened_preds)

# Compute F1 score
f1 = f1_score(flattened_labels, flattened_preds, average="weighted")

# Print the classification report
print("Classification Report:")
print(classification_report(flattened_labels, flattened_preds, digits=4))

# Print overall accuracy
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score: {f1:.4f}")


# %% [markdown]
# ### Testing

# %%
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Prepare the test data loader
test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=collate_fn)

# Evaluate on the test set
model.eval()
test_predictions, test_true_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            test_predictions.append([label_list[p] for p, l in zip(pred, label) if l != -100])
            test_true_labels.append([label_list[l] for p, l in zip(pred, label) if l != -100])

# Flatten predictions and true labels for the test set
flattened_test_preds = [label for seq in test_predictions for label in seq]
flattened_test_labels = [label for seq in test_true_labels for label in seq]

# Compute metrics
test_accuracy = accuracy_score(flattened_test_labels, flattened_test_preds)
test_f1 = f1_score(flattened_test_labels, flattened_test_preds, average="weighted")

# Print the complete classification report for the test set
print("Test Classification Report:")
print(classification_report(flattened_test_labels, flattened_test_preds, digits=4))

# Print test accuracy and F1 score
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")


# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(flattened_test_labels, flattened_test_preds, output_dict=True)

# Extract class-wise F1-scores
classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
f1_scores = [report[cls]["f1-score"] for cls in classes]

# Bar plot for F1-scores
plt.figure(figsize=(8, 6))
plt.bar(classes, f1_scores, color='skyblue')
plt.xlabel("Classes")
plt.ylabel("F1 Score")
plt.title("F1 Score by Class")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.grid(axis="y")
plt.show()


# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Compute confusion matrix
labels = list(set(flattened_test_labels))
cm = confusion_matrix(flattened_test_labels, flattened_test_preds, labels=labels)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical")
plt.title("Confusion Matrix for NER Tags")
plt.show()


# %%
# Define a test sentence
test_sentence = "Barack Obama visited Paris."

inputs = tokenizer(
    test_sentence,
    return_tensors="pt",
    truncation=True,
    padding=True,
    is_split_into_words=False
)

# Remove token_type_ids if present
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# Move inputs to the appropriate device
inputs = {key: val.to(device) for key, val in inputs.items()}

# Run the model
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions
logits = outputs["logits"]
predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

# Decode predictions back to labels
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu().numpy())
predicted_labels = [label_list[pred] for pred in predictions]

# Print the input and output
print("Test Input Sentence:")
print(test_sentence)
print("\nTokenized Input:")
print(tokens)
print("\nPredicted Labels:")
print(predicted_labels)


# %% [markdown]
# ### Model Architecture

# %%
import os
import torch

# Define the directory to save the model
save_directory = "./bert_ner_model"
os.makedirs(save_directory, exist_ok=True)

# Save the model
torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

# Save model configuration
with open(os.path.join(save_directory, "config.json"), "w") as f:
    f.write(model.bert.config.to_json_string())

print(f"Model, tokenizer, and configuration saved to {save_directory}")


# %%
from torchviz import make_dot
from transformers import AutoTokenizer

# Create dummy inputs for the model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dummy_sentence = "Steve Jobs founded Apple in California."
dummy_inputs = tokenizer(dummy_sentence, return_tensors="pt", padding=True, truncation=True)

# Move inputs to the appropriate device
dummy_inputs = {key: val.to(device) for key, val in dummy_inputs.items()}

# Forward pass through the model
model.eval()
outputs = model(input_ids=dummy_inputs["input_ids"], attention_mask=dummy_inputs["attention_mask"])

# Visualize the computation graph
graph = make_dot(outputs["logits"], params=dict(model.named_parameters()))
graph.render("model_architecture", format="png")

print("Model architecture diagram saved as 'model_architecture.png'.")


# %%
from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='Enhanced BERT NER Model Architecture', format="png")

# Add nodes
dot.node('A', 'Input Sentence\n(e.g., "Steve Jobs founded Apple.")', shape='box')
dot.node('B', 'Tokenizer\n(WordPiece Tokenization)', shape='box')
dot.node('C', 'BERT-base Encoder\n(Hidden States for All Layers)', shape='box')
dot.node('D', 'Layer Attention\n(Combine All Layers)', shape='ellipse')
dot.node('E', 'Dropout Layer\n(Regularization)', shape='ellipse')
dot.node('F', 'Fully Connected Layer\n(Classifier)', shape='ellipse')
dot.node('G', 'Output Tags\n(e.g., [B-PER, O, B-ORG])', shape='box')

# Connect nodes
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# Render the graph
dot.render('enhanced_bert_ner_model', format="png", cleanup=True)

print("Enhanced model diagram saved as 'enhanced_bert_ner_model.png'.")


# %%



