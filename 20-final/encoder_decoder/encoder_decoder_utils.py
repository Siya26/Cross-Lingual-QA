from datasets import load_dataset
import tqdm
import torch
import json
from transformers import GenerationConfig

def load_data(is_squad):
    if is_squad:
        dataset = load_dataset("squad")
        return dataset["train"], dataset["validation"]
    else:
        with open("../combine_train.json", 'r') as file:
            train_data = json.load(file)
        with open("../combine_dev.json", 'r') as file:
            val_data = json.load(file)
        return train_data, val_data

def load_model(model_name):
    if model_name == "t5":
        from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
        model_name = "google/mt5-base"
        tokenizer = MT5TokenizerFast.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        print("Loaded T5 model")
    elif model_name == "bart":
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model_name = "facebook/mbart-large-50"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)

    return model, tokenizer 
    

def encode_data(batch, tokenizer, is_squad=True):
    inputs = []
    if is_squad:
        answers = batch["answers"]
        contexts = batch["context"]
        questions = batch["question"]
    else:
        contexts = [example["context"] for example in batch]
        questions = [example["question"] for example in batch]
        answers = [example["answer_text"] for example in batch]

    inputs_encoded = tokenizer(questions, contexts, padding="max_length", truncation=True, return_tensors="pt", max_length=256, add_special_tokens=True)

    if is_squad:
        labels = tokenizer(
            [ans["text"][0] if ans["text"] else "" for ans in answers],
            padding="max_length", truncation=True, return_tensors="pt", max_length=256, add_special_tokens=True
        )["input_ids"]
    else:
        labels = tokenizer(
            answers, 
            padding="max_length", truncation=True, return_tensors="pt", max_length=256, add_special_tokens=True
        )["input_ids"]

    return inputs_encoded["input_ids"],inputs_encoded["attention_mask"],labels

def train(model, train_data, val_data, batch_size, tokenizer, optimizer, epochs, device, is_squad=True):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        train_loss = 0.0
        val_loss = 0.0

        train_steps = 0
        val_steps = 0

        model.train()

        for i in tqdm.tqdm(range(0, len(train_data), batch_size), desc="Training Batches"):
            
            batch = train_data[i:i+batch_size]

            optimizer.zero_grad()
            input_ids, attention_masks, labels = encode_data(batch, tokenizer, is_squad)
            
            # input_ids = torch.tensor(input_ids)
            # attention_masks = torch.tensor(attention_masks)
            # labels = torch.tensor(labels)

            # input_ids = input_ids.to(device)
            # attention_masks = attention_masks.to(device)
            # labels = labels.to(device)

            input_ids = input_ids.clone().detach().to(device)
            attention_masks = attention_masks.clone().detach().to(device)
            labels = labels.clone().detach().to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        model.eval()

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(val_data), batch_size), desc="Validation Batches"):
                batch = val_data[i:i+batch_size]
                input_ids, attention_masks, labels = encode_data(batch, tokenizer, is_squad)

                input_ids = torch.tensor(input_ids)
                attention_masks = torch.tensor(attention_masks)
                labels = torch.tensor(labels)

                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()
                val_steps += 1

        train_losses.append(train_loss / train_steps)
        val_losses.append(val_loss / val_steps)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]} - Val Loss: {val_losses[-1]}")

    return train_losses, val_losses, model

def encode_test_data(context, question, tokenizer, max_length=256):
    inputs = tokenizer(question, context, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length, add_special_tokens=True)
    return inputs, inputs['input_ids'], inputs['attention_mask']

def test_model(model, test_data, batch_size, tokenizer, device, max_length=512):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]

            context = []
            question = []
            for j in range(len(batch)):
                context.append(batch[j]['context'])
                question.append(batch[j]['question'])
            
            inputs, input_ids, attention_masks = encode_test_data(context, question, tokenizer)
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=max_length,
            )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_outputs)
    
    return predictions

