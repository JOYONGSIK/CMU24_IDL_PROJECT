def test_epoch(model, data_loader, device):
    model.eval()
    predictions, gold = [], []
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, acoustic_input=acoustic_input, visual_input=visual_input, labels=labels)
            logits = outputs['logits']
            pred = logits.argmax(dim=-1)
            predictions.extend(pred.tolist())
            gold.extend(labels.tolist())
            correct += int((pred == labels).sum())
    return correct / len(data_loader.dataset), predictions, gold