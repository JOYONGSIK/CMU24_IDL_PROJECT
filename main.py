import torch
from torch.utils.data import DataLoader
from models.multimodal_bart import MultimodalBartForSequenceClassification
from dataset import MultimodalSarcasmDataset
from train import train_epoch
from test import test_epoch
from data_utils import pad_seq, load_audio, load_frames
from transformers import BartTokenizerFast, BartConfig
import os

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
AUDIO_DIR = "./data/audio"
FRAMES_DIR = "./data/frames"
MODEL_DIR = "./saved_models"

# Dataset paths
TRAIN_DATA_PATH = "./data/train.pkl"
VALID_DATA_PATH = "./data/valid.pkl"
TEST_DATA_PATH = "./data/test.pkl"

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)

# Tokenizer and model initialization
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
config = BartConfig.from_pretrained("facebook/bart-base")
model = MultimodalBartForSequenceClassification(config=config).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load datasets
train_dataset = MultimodalSarcasmDataset.load_from_file(TRAIN_DATA_PATH, AUDIO_DIR, FRAMES_DIR, tokenizer)
valid_dataset = MultimodalSarcasmDataset.load_from_file(VALID_DATA_PATH, AUDIO_DIR, FRAMES_DIR, tokenizer)
test_dataset = MultimodalSarcasmDataset.load_from_file(TEST_DATA_PATH, AUDIO_DIR, FRAMES_DIR, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training and validation
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    valid_accuracy, valid_predictions, valid_labels = test_epoch(model, valid_loader, DEVICE)
    print(f"Validation Accuracy: {valid_accuracy:.4f}")

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        model_path = os.path.join(MODEL_DIR, f"best_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model to {model_path}")

# Testing
model_path = os.path.join(MODEL_DIR, "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)

test_accuracy, test_predictions, test_labels = test_epoch(model, test_loader, DEVICE)
print(f"Test Accuracy: {test_accuracy:.4f}")