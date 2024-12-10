import torch
from tqdm import tqdm

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc='Training')):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    return epoch_train_loss