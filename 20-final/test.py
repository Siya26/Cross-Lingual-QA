import torch
import sys
import json
from collections import Counter
import string
import re
import json
import sys
import unicodedata
from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
MIXED_SEGMENTATION_LANGS = ['zh']

def whitespace_tokenize(text):
    return text.split()

def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text # Hindi does not have formal articles
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'zh':
            return text # Chinese does not have formal articles
        else:
            raise Exception('Unknown Language {}'.format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)

def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth, lang):
    return (normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang))

def rougeScore(prediction, ground_truth, lang):
    ground_truth = normalize_answer(ground_truth, lang)
    prediction = normalize_answer(prediction, lang)
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure

# def bleuScore(prediction, ground_truth, lang):
#     ground_truth = normalize_answer(ground_truth, lang)
#     prediction = normalize_answer(prediction, lang)
#     return sentence_bleu([ground_truth.split()], prediction.split())
    
def bertScore(predictions, ground_truths, lang):
    predictions = [normalize_answer(x, lang) for x in predictions]
    ground_truths = [normalize_answer(x, lang) for x in ground_truths]
    P, R, F1 = score(predictions, ground_truths, lang=lang, model_type='bert-base-multilingual-cased')
    return F1.mean().item()

def evaluate_model(test_data, predictions, lang):
    actual_answers = [x['answer_text'] for x in test_data]
    print(actual_answers[:5])
    f1 = exact_match = total = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        ground_truth = actual_answers[i]
        total += 1
        exact_match += exact_match_score(prediction, ground_truth, lang)
        f1 += f1_score(prediction, ground_truth, lang)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    print(f"Exact Match: {exact_match:.2f}")
    print(f"F1: {f1:.2f}")

def evaluate_generative_model(test_data, predictions, lang):
    actual_answers = [x['answer_text'] for x in test_data]
    print(actual_answers[:5])
    
    rouge_f1 = total = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        ground_truth = actual_answers[i]
        total += 1
        rouge_f1 += rougeScore(prediction, ground_truth, lang)

    rouge_f1 = rouge_f1 / total
    bertscore = bertScore(predictions, actual_answers, lang)

    print(f"Rouge F1: {rouge_f1:.2f}")
    print(f"BertScore: {bertscore:.2f}")     

if __name__ == "__main__":

    model_type = sys.argv[1]
    model_name = sys.argv[2]
    model_file = sys.argv[3]
    context_lang = sys.argv[4]
    question_lang = sys.argv[5]

    test_file = f"MLQA_data/test-context-{context_lang}-question-{question_lang}.json"

    with open(test_file, 'r') as file:
        data = json.load(file)
            
    model = torch.load(model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 32

    if model_type == '1':
        from encoder.encoder_utils import test_model, load_model
        _,tokenizer = load_model(model_name)
        predictions = test_model(model_name, model, data, batch_size, tokenizer, device)
        evaluate_model(data, predictions, context_lang)
    elif model_type == '2':
        from encoder_decoder.encoder_decoder_utils import test_model, load_model
        _,tokenizer = load_model(model_name)
        predictions = test_model(model, data, batch_size, tokenizer, device)
        evaluate_generative_model(data, predictions, context_lang)
    else:
        from decoder.decoder_utils import test_model, load_model
        _,tokenizer = load_model(model_name)
        predictions = test_model(model, data, batch_size, tokenizer, device)
        evaluate_generative_model(data, predictions, context_lang)

    print(predictions[:5])
