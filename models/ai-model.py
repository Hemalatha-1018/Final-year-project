# Import necessary libraries
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

# Download NLTK data
nltk.download('punkt')

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Load Indic-BERT for Indian language support
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_bert = AutoModel.from_pretrained("ai4bharat/indic-bert")

# Step 1: Load the dataset
def load_schemes():
    with open('scheme1.json', 'r') as file:
        schemes = json.load(file)
    return schemes

# Step 2: NLP Pipeline
# 2.1: Language Detection
def detect_language(text):
    try:
        lang = detect(text)
        return lang  # Returns 'en' for English, 'ta' for Tamil, 'hi' for Hindi, etc.
    except:
        return 'en'  # Default to English if detection fails

# 2.2: Entity Extraction using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = {"income": None, "location": None, "occupation": None}
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["income"] = ent.text
        elif ent.label_ == "GPE":
            entities["location"] = ent.text
        elif ent.label_ in ["ORG", "PERSON"]:
            if "farmer" in ent.text.lower() or "student" in ent.text.lower():
                entities["occupation"] = ent.text.lower()
    return entities

# 2.3: Sentiment Analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment < -0.1:
        return "negative"  # For grievance escalation
    elif sentiment > 0.1:
        return "positive"
    else:
        return "neutral"

# Process text with Indic-BERT
def process_text_with_indic_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = indic_bert(**inputs)
    return outputs.last_hidden_state  # Returns embeddings for the text

# Step 4: Rule-Based Engine for Eligibility Checks
def check_eligibility(user_input, entities, schemes):
    matched_schemes = []
    user_input = user_input.lower()

    # Keywords for categories
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

    # Additional eligibility checks using entities
    if entities["occupation"] or entities["location"] or entities["income"]:
        filtered_schemes = []
        for scheme in matched_schemes:
            eligibility = scheme.get("eligibility", {})
            match = True
            if entities["occupation"] and eligibility.get("occupation"):
                if entities["occupation"] not in str(eligibility["occupation"]).lower():
                    match = False
            if entities["location"] and eligibility.get("location"):
                if entities["location"].lower() not in str(eligibility["location"]).lower():
                    match = False
            if entities["income"] and eligibility.get("income"):
                if entities["income"] not in str(eligibility["income"]).lower():
                    match = False
            if match:
                filtered_schemes.append(scheme)
        return filtered_schemes
    return matched_schemes

# Step 5: Main Chatbot Function with Messaging-Style Output
def chatbot():
    print("🌟 Welcome to the Advanced Government Scheme Chatbot 2.0! 🌟")
    print("I’m here to help you find government schemes for categories like farmers, health, women, education, housing, or startups.")
    print("Just tell me what you need, or type 'exit' to quit.\n")

    # Load schemes
    schemes = load_schemes()
    
    # Metrics tracking
    start_time = time.time()

    while True:
        user_input = input("💬 You: ")
        
        if user_input.lower() == "exit":
            print("🤖 Chatbot: Goodbye! If you need help later, just come back! 👋")
            break

        # Measure latency
        query_start = time.time()

        # NLP Pipeline
        lang = detect_language(user_input)
        entities = extract_entities(user_input)
        sentiment = analyze_sentiment(user_input)

        # Rule-Based Eligibility Check
        matched_schemes = check_eligibility(user_input, entities, schemes)

        # Display results in a messaging style
        if not matched_schemes or isinstance(matched_schemes[0], str):
            print(f"🤖 Chatbot: Oh no! 😔 I couldn’t find any schemes for your request.")
            print(f"🤖 Chatbot: Try using keywords like 'farmer', 'health', 'women', 'education', 'housing', or 'startup'. What else can I help with?")
        else:
            print(f"🤖 Chatbot: Great news! I found {len(matched_schemes)} schemes that might help you. Let me share them one by one. 😊")

            for idx, scheme in enumerate(matched_schemes, 1):
                # Handle scheme name if it's a dictionary
                scheme_name = scheme.get('name', 'N/A')
                if isinstance(scheme_name, dict):
                    scheme_name = scheme_name.get(lang, scheme_name.get('en', 'N/A'))  # Use detected language or default to English

                # Present the scheme in a conversational way
                print(f"🤖 Chatbot: Here’s scheme number {idx}: *{scheme_name}*")
                print(f"🤖 Chatbot: It’s about {scheme.get('description', 'helping people in this category (no detailed description available)')}. You can get {scheme.get('benefits', 'some benefits (not specified)')}, which sounds pretty useful!")
                print(f"🤖 Chatbot: To be eligible, you need: {scheme.get('eligibility', 'some criteria (not specified)')}\n")

        # Metrics: Latency
        query_end = time.time()
        latency = query_end - query_start
        print(f"🤖 Chatbot: ⏱️ I responded in {latency:.2f} seconds.\n")

    # Final Metrics
    total_time = time.time() - start_time
    print("🤖 Chatbot: 📊 Final Session Metrics:")
    print(f"   - Total Session Time: {total_time:.2f} seconds")