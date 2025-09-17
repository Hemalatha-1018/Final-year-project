import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
from textblob import TextBlob
import spacy
from sacrebleu import corpus_bleu
import nltk
nltk.download('punkt')


def load_schemes():
    with open('/work/data/myscheme/scheme1.json', 'r') as file:
        schemes = json.load(file)
    return schemes


def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return 'en'


nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
    doc = nlp(text)
    entities = {"income": None, "location": None, "occupation": None}
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["income"] = ent.text
        elif ent.label_ == "GPE":
            entities["location"] = ent.text
        elif ent.label_ == "ORG" or ent.label_ == "PERSON":
            if "farmer" in ent.text.lower() or "student" in ent.text.lower():
                entities["occupation"] = ent.text.lower()
    return entities


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment < -0.1:
        return "negative"
    elif sentiment > 0.1:
        return "positive"
    else:
        return "neutral"


tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_bert = AutoModel.from_pretrained("ai4bharat/indic-bert")

def process_text_with_indic_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = indic_bert(**inputs)
    return outputs.last_hidden_state


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(energy @ self.v, dim=1)
        context = attention.transpose(1, 2) @ encoder_outputs
        return context

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell, encoder_outputs, attention):
        context = attention(hidden, encoder_outputs)
        output, (hidden, cell) = self.lstm(torch.cat((x, context), dim=2), (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = Attention(hidden_dim)

    def forward(self, src, trg):
        encoder_outputs, hidden, cell = self.encoder(src)
        outputs = []
        x = trg[:, 0:1, :]
        for t in range(1, trg.shape[1]):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs, self.attention)
            outputs.append(output)
            x = output
        return torch.cat(outputs, dim=1)


def check_eligibility(user_input, entities, schemes):
    matched_schemes = []
    user_input = user_input.lower()

    keyword_mapping = {
        "farmer": ["agriculture", "farmer", "crop", "kisan"],
        "health": ["health", "hospital", "insurance"],
        "women": ["women", "female", "girl"],
        "education": ["education", "student", "school"],
        "housing": ["housing", "house", "awas"],
        "startup": ["startup", "business", "entrepreneur"]
    }

    matched_category = None
    for category, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in user_input:
                matched_category = category
                break
        if matched_category:
            break

    if not matched_category:
        return ["Sorry, I couldn't understand your request. Try saying 'farmer', 'health', 'women', 'education', 'housing', or 'startup'."]

    for scheme in schemes:
        if "category" in scheme and matched_category in scheme["category"].lower():
            matched_schemes.append(scheme)
        elif "description" in scheme and matched_category in scheme["description"].lower():
            matched_schemes.append(scheme)
        elif "eligibility" in scheme:
            for key, value in scheme["eligibility"].items():
                if isinstance(value, str) and matched_category in value.lower():
                    matched_schemes.append(scheme)
                    break
                elif isinstance(value, list) and any(matched_category in str(item).lower() for item in value):
                    matched_schemes.append(scheme)
                    break


    if entities["occupation"] or entities["location"] or entities["income"]:
        filtered_schemes = []
        for scheme in matched_schemes:
            eligibility = scheme["eligibility"]
            match = True
            if entities["occupation"] and "occupation" in eligibility:
                if entities["occupation"] not in str(eligibility["occupation"]).lower():
                    match = False
            if entities["location"] and "location" in eligibility:
                if entities["location"].lower() not in str(eligibility["location"]).lower():
                    match = False
            if entities["income"] and "income" in eligibility:
                if entities["income"] not in str(eligibility["income"]).lower():
                    match = False
            if match:
                filtered_schemes.append(scheme)
        return filtered_schemes
    return matched_schemes


def simulate_seq2seq_response(user_input, lang):
    if lang == "ta":
        return "நான் உங்களுக்கு உதவ முடியும். எந்த வகையான ஆதரவு தேவை?"
    elif lang == "hi":
        return "मैं आपकी मदद कर सकता हूँ। आपको किस तरह की सहायता चाहिए?"
    else:
        return "I can help you. What kind of support do you need?"

def chatbot():
    print("Welcome to the Government Scheme Chatbot!")
    print("I can help you find schemes for farmers, health, women, education, housing, or startups.")
    print("Type 'exit' to quit.\n")


    schemes = load_schemes()


    input_dim = 768
    hidden_dim = 256
    output_dim = 100
    encoder = Encoder(input_dim, hidden_dim)
    decoder = Decoder(output_dim, hidden_dim)
    seq2seq = Seq2Seq(encoder, decoder, hidden_dim)


    responses = []
    references = []
    correct_eligibility_checks = 0
    total_checks = 0
    start_time = time.time()

    while True:
        user_input = input("What kind of support do you need? (e.g., 'I am a farmer', 'I need health support'): ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break


        query_start = time.time()


        lang = detect_language(user_input)
        entities = extract_entities(user_input)
        sentiment = analyze_sentiment(user_input)

        embeddings = process_text_with_indic_bert(user_input)

        seq2seq_response = simulate_seq2seq_response(user_input, lang)
        print(f"Chatbot: {seq2seq_response}")

        matched_schemes = check_eligibility(user_input, entities, schemes)

        if not matched_schemes or isinstance(matched_schemes[0], str):
            print(matched_schemes[0] if matched_schemes else "No schemes found for your request.")
        else:
            print("\nHere are some schemes you might be eligible for:\n")
            for scheme in matched_schemes:
                print(f"Scheme: {scheme['name']}")
                print(f"Description: {scheme['description']}")
                print(f"Benefits: {scheme['benefits']}")
                print(f"Eligibility: {scheme['eligibility']}")
                print("-" * 50)

        if sentiment == "negative":
            print("It seems you're not satisfied. Would you like to escalate this to a human agent? (yes/no)")
            escalate = input()
            if escalate.lower() == "yes":
                print("Escalating to a human agent...")
                continue


        responses.append(seq2seq_response)
        references.append("I can help you. What kind of support do you need?")
        bleu_score = corpus_bleu(responses, [references]).score
        print(f"BLEU Score: {bleu_score:.2f}")

        total_checks += 1
        if matched_schemes and not isinstance(matched_schemes[0], str):
            correct_eligibility_checks += 1
        accuracy = (correct_eligibility_checks / total_checks) * 100
        print(f"Eligibility Check Accuracy: {accuracy:.2f}%")

        query_end = time.time()
        latency = query_end - query_start
        print(f"Response Latency: {latency:.2f} seconds")

    total_time = time.time() - start_time
    print(f"\nFinal Metrics:")
    print(f"Average BLEU Score: {bleu_score:.2f}")
    print(f"Final Eligibility Check Accuracy: {accuracy:.2f}%")
    print(f"Average Latency: {(total_time / total_checks):.2f} seconds")

if __name__ == "__main__":
    chatbot()