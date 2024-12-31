from datasets import load_dataset
import tqdm
import torch
import json

def load_data(is_squad):
    if is_squad:
        dataset = load_dataset("squad")
        return dataset["train"], dataset["validation"]
    else:
        with open("combine_train.json", 'r') as file:
            train_data = json.load(file)
        with open("combine_dev.json", 'r') as file:
            val_data = json.load(file)
        return train_data, val_data
    
def load_model(model_name):
    if model_name == "bert":
        from transformers import BertTokenizerFast, BertForQuestionAnswering
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
        model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-uncased")
    else:
        from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")
    return model, tokenizer
        

def encode_data(batch, tokenizer,model_name, is_squad=True):
    
    if is_squad:
        ans = batch["answers"]
        inputs = tokenizer(batch["context"], batch["question"], padding=True, truncation=True)
        n = len(ans)
    else:
        contexts = [example["context"] for example in batch]
        questions = [example["question"] for example in batch]

        inputs = tokenizer(contexts, questions, padding=True, truncation=True)
        n = len(batch)

    start_positions = []
    end_positions = []

    for i in range(n):
        if is_squad:
            data = ans[i]
            if(len(data["answer_start"]) == 0):
                start_positions.append(0)
                end_positions.append(0)
                continue
            start_token = inputs.char_to_token(i, data["answer_start"][0])
            end_token = inputs.char_to_token(i, data["answer_start"][0] + len(data["text"][0]) -1)
        else:
            data = batch[i]
            start_token = inputs.char_to_token(i, data["answer_start"])
            end_token = inputs.char_to_token(i, data["answer_start"] + len(data["answer_text"]) -1)

        if start_token is None:
            start_token = tokenizer.model_max_length
        if end_token is None:
            end_token = tokenizer.model_max_length

        start_positions.append(start_token)
        end_positions.append(end_token)

    input_ids = inputs["input_ids"]
    if model_name == "bert":
        segment_ids = inputs["token_type_ids"]
    attention_masks = inputs["attention_mask"]

    if model_name == "bert":
        return start_positions, end_positions, input_ids, segment_ids, attention_masks
    return start_positions, end_positions, input_ids, attention_masks

def train(model, model_name,train_data, val_data, batch_size, tokenizer, optimizer, epochs, device, is_squad=True):
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
            if model_name == "bert":
                start_positions, end_positions, input_ids, segment_ids, attention_masks = encode_data(batch, tokenizer, model_name, is_squad)
            else:
                start_positions, end_positions, input_ids, attention_masks = encode_data(batch, tokenizer, model_name,is_squad)

            input_ids = torch.tensor(input_ids)
            if model_name == "bert":
                segment_ids = torch.tensor(segment_ids)
            attention_masks = torch.tensor(attention_masks)
            start_positions = torch.tensor(start_positions)
            end_positions = torch.tensor(end_positions)

            input_ids = input_ids.to(device)
            if model_name == "bert":
                segment_ids = segment_ids.to(device)
            attention_masks = attention_masks.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            if model_name == "bert":
                outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_masks, start_positions=start_positions, end_positions=end_positions)
            else:
                outputs = model(input_ids, attention_mask=attention_masks, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        model.eval()

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(val_data), batch_size), desc="Validation Batches"):
                batch = val_data[i:i+batch_size]
                if model_name == "bert":
                    start_positions, end_positions, input_ids, segment_ids, attention_masks = encode_data(batch, tokenizer, model_name, is_squad)
                else:
                    start_positions, end_positions, input_ids, attention_masks = encode_data(batch, tokenizer, model_name, is_squad)

                input_ids = torch.tensor(input_ids)
                if model_name == "bert":
                    segment_ids = torch.tensor(segment_ids)
                attention_masks = torch.tensor(attention_masks)
                start_positions = torch.tensor(start_positions)
                end_positions = torch.tensor(end_positions)
                
                input_ids = input_ids.to(device)
                if model_name == "bert":
                    segment_ids = segment_ids.to(device)
                attention_masks = attention_masks.to(device)
                start_positions = start_positions.to(device)
                end_positions = end_positions.to(device)

                if model_name == "bert":
                    outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_masks, start_positions=start_positions, end_positions=end_positions)
                else:
                    outputs = model(input_ids, attention_mask=attention_masks, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss

                val_loss += loss.item()
                val_steps += 1

        train_losses.append(train_loss / train_steps)
        val_losses.append(val_loss / val_steps)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]} - Val Loss: {val_losses[-1]}")

    return train_losses, val_losses, model

def encode_test_data(model_name, context, question, tokenizer):
    inputs = tokenizer(context, question, padding=True, truncation=True, return_offsets_mapping=True)
    if model_name == "bert":
        return inputs, inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'] 
    else:
        return inputs, inputs['input_ids'], inputs['attention_mask']
    
def test_model(model_name, model, test_data, batch_size, tokenizer, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]

            context = []
            question = []
            for i in range(len(batch)):
                context.append(batch[i]['context'])
                question.append(batch[i]['question'])

            
            if model_name == "bert":
                inputs, input_ids, segment_ids, attention_masks = encode_test_data(model_name,context, question, tokenizer)
            else:
                inputs, input_ids, attention_masks = encode_test_data(model_name,context, question, tokenizer)
            
            input_ids = torch.tensor(input_ids)
            if model_name == "bert":
                segment_ids = torch.tensor(segment_ids)
            attention_masks = torch.tensor(attention_masks)
            
            input_ids = input_ids.to(device)
            if model_name == "bert":
                segment_ids = segment_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            if model_name == "bert":
                outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=segment_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_masks)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_probs = torch.nn.functional.softmax(start_logits, dim=1)
            end_probs = torch.nn.functional.softmax(end_logits, dim=1)

            start_pred = torch.argmax(start_probs, dim=1)
            end_pred = torch.argmax(end_probs, dim=1)

            for j in range(len(batch)):
                offsets = inputs['offset_mapping'][j]
                start_idx = start_pred[j].item()
                end_idx = end_pred[j].item()
                start_char = offsets[start_idx][0]
                end_char = offsets[end_idx][1]
                answer = context[j][start_char:end_char]
                predictions.append(answer)
            
    return predictions