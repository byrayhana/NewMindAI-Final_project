import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import WordNetLemmatizer
import os

nltk.download('stopwords')
stops = set(stopwords.words("english"))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cpu")


# Verilen texti temizleme fonksiyonu
def clean_text(text, remove_stopwords=True):
    text = re.sub(r'\[NAME\]', '', text)
    text = text.lower()
    text = text.split()
    new_text = []
    for word in text:
        word = word.replace("’", "'")
        new_text.append(word)

    text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\d+', '', text)
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text
def clean_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    text = clean_text(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
# BERT modelini yükleme
def load_bert_model(model_path):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model
def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


# BERT model yükleme
bert_path = "best_model.pt"
bert_model = load_bert_model(bert_path)
bert_tokenizer = load_tokenizer()
categories = {0: 'negative', 1: 'neutral', 2: 'positive', 3: 'mixed'}

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 Tokenizer ve Modelini Yükleme
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_conclusion(input_text, predicted_category):
    # Pad token
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    if predicted_category == 'positive':
        prompt = (
            f"Given that the text has a positive sentiment, summarize the following in an encouraging and uplifting tone:\n\n"
            f"{input_text}\n\n"
            "Conclusion:")
    elif predicted_category == 'negative':
        prompt = (
            f"Given that the text has a negative sentiment, summarize the following in a supportive and comforting tone:\n\n"
            f"{input_text}\n\n"
            "Conclusion:")
    elif predicted_category == 'neutral':
        prompt = (
            f"Given that the text has a neutral tone, summarize the following in an unbiased and factual manner:\n\n"
            f"{input_text}\n\n"
            "Conclusion:")
    elif predicted_category == 'mixed':
        prompt = (
            f"Given that the text has a mixed sentiment, summarize the following in a balanced and objective tone, considering both aspects:\n\n"
            f"{input_text}\n\n"
            "Conclusion:")
    else:
        prompt = f"Summarize the following text:\n\n{input_text}\n\nConclusion:"

    # Tokenize the input
    inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate the output with no repetition of n-grams
    outputs = gpt2_model.generate(
        inputs['input_ids'],
        max_length=200,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.8 # Control randomness
    )

    # Decode the output into text
    conclusion = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The conclusion will follow the word "Conclusion:" in the output
    conclusion = conclusion.split('Conclusion:')[-1].strip()

    return conclusion



# Streamlit UI
st.title("Emotion Classification and Conclusion")

input_text = st.text_area("Input Text", height=200)

if st.button("Classify"):
    if not input_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        words = input_text.split()
        processed_text = clean_and_lemmatize(input_text)
        processed_text = processed_text[:512]

        with torch.no_grad():
            inputs = bert_tokenizer(processed_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            outputs = bert_model(input_ids, attention_mask=attention_mask)

        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        bert_category = categories[predicted_class]

        st.success(f"BERT model classification: '{bert_category}'")

        conclusion = generate_conclusion(input_text, bert_category)
        st.success(f"**Generated Conclusion:** {conclusion}")
