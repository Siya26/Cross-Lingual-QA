from datasets import load_dataset
import tqdm
import torch
import json

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
    if model_name == "gpt":
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model_name = "gpt2"  
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif model_name == "opt":
        from transformers import OPTForCausalLM, AutoTokenizer
        model_name = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

    if is_squad:
        inputs = [f"Context: {context} Question: {question} Answer: {ans['text'][0] if ans['text'] else ''}" 
                  for question, context, ans in zip(questions, contexts, answers)]
    else:
        inputs = [f"Context: {context} Question: {question} Answer: {answer}" 
                  for question, context, answer in zip(questions, contexts, answers)]

    inputs_encoded = tokenizer(inputs, padding='max_length', truncation=True, return_tensors="pt", max_length=1024)
    labels_encoded = inputs_encoded["input_ids"].clone()
    labels_encoded[labels_encoded == tokenizer.pad_token_id] = -100

    return inputs_encoded["input_ids"], inputs_encoded["attention_mask"], labels_encoded

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

                val_loss += loss.item()
                val_steps += 1

        train_losses.append(train_loss / train_steps)
        val_losses.append(val_loss / val_steps)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]} - Val Loss: {val_losses[-1]}")

    return train_losses, val_losses, model

def encode_test_data(context, question, tokenizer, max_length=512):
    inputs = [f"Context: {context[i]} Question: {question[i]} Answer: " for i in range(len(context))]
    
    tokenized = tokenizer(
        inputs, 
        padding="max_length",
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    return tokenized, tokenized["input_ids"], tokenized["attention_mask"]

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
                max_new_tokens=50,
            )

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            cleaned_outputs = []
            for output in decoded_outputs:
                ans = output.split("Answer:")
                if len(ans) > 1:
                    cleaned_outputs.append(ans[1].strip())
                else:
                    cleaned_outputs.append("")
            predictions.extend(cleaned_outputs)
    
    return predictions

