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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Check PyTorch version and provide detailed feedback
print(f"Current PyTorch version: {torch.__version__}")
if torch.__version__ < '2.0.0':
    raise Exception(
        f"PyTorch version {torch.__version__} is outdated. Upgrade to 2.0.0 or later.\n"
        "Run these commands in your terminal:\n"
        "1. pip install --upgrade torch\n"
        "2. If using torchvision, upgrade it too: pip install --upgrade torchvision\n"
        "3. Verify with: python -c 'import torch; print(torch.__version__)'\n"
        "Ensure you're in the correct Python environment (e.g., activate your virtualenv)."
    )

# Download NLTK data
nltk.download('punkt')

# Download and load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load Indic-BERT with proper tokenizer handling
try:
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
    indic_bert = AutoModel.from_pretrained("ai4bharat/indic-bert")
except Exception as e:
    print(f"Failed to load Indic-BERT: {e}. Falling back to BERT.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=False)
    indic_bert = AutoModel.from_pretrained("bert-base-multilingual-cased")

# Load a QA pipeline for detailed scheme explanations with error handling
try:
    from transformers import pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
except ImportError as e:
    print(f"Failed to load QA pipeline: {e}. Detailed answers will be unavailable.")
    qa_pipeline = None

# Step 1: Load the dataset
def load_schemes():
    try:
        with open('/work/data/myscheme/scheme1.json', 'r') as file:
            schemes = json.load(file)
        return schemes
    except FileNotFoundError:
        print("Error: 'scheme1.json' file not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'scheme1.json'.")
        return []

# Step 2: Advanced NLP Pipeline
# 2.1: Language Detection
def detect_language(text):
    return detect(text) if text else 'en'

# 2.2: Enhanced Entity Extraction
def extract_entities(text):
    doc = nlp(text)
    entities = {"income": None, "location": None, "occupation": None, "category": None}
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            entities["income"] = ent.text
        elif ent.label_ == "GPE":
            entities["location"] = ent.text
        elif ent.label_ in ["ORG", "PERSON"]:
            if "farmer" in ent.text.lower() or "student" in ent.text.lower():
                entities["occupation"] = ent.text.lower()
    # Extract category from keywords
    categories = ["farmer", "health", "women", "education", "housing", "startup"]
    for cat in categories:
        if cat in text.lower():
            entities["category"] = cat
            break
    return entities

# 2.3: Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "negative" if sentiment < -0.1 else "positive" if sentiment > 0.1 else "neutral"

# 2.4: Intent Recognition with TF-IDF and Cosine Similarity
def recognize_intent(user_input, schemes):
    predefined_intents = {
        "scheme_info": ["tell me about schemes", "what schemes are available", "list schemes"],
        "eligibility": ["am i eligible", "who qualifies", "eligibility criteria"],
        "details": ["more details", "explain scheme", "what does it offer"]
    }
    all_texts = [user_input.lower()] + [intent for intents in predefined_intents.values() for intent in intents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    max_idx = np.argmax(similarity)
    intent = list(predefined_intents.keys())[max_idx // len(list(predefined_intents.values())[0])]
    return intent

# 2.5: Process text with Indic-BERT for embeddings
def process_text_with_indic_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = indic_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Step 3: Advanced Eligibility Check with Contextual Understanding
def check_eligibility(user_input, entities, schemes, context=None):
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

    matched_category = entities.get("category")
    if not matched_category:
        for category, keywords in keyword_mapping.items():
            if any(keyword in user_input for keyword in keywords):
                matched_category = category
                break

    if not matched_category:
        return ["Sorry, I couldn't understand your request. Try keywords like 'farmer', 'health', etc."], None

    for scheme in schemes:
        if "category" in scheme and matched_category in scheme["category"].lower():
            matched_schemes.append(scheme)
        elif "description" in scheme and matched_category in scheme["description"].lower():
            matched_schemes.append(scheme)

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

    return filtered_schemes, matched_category

# Step 4: QA Module for Detailed Explanations
def answer_question(question, scheme):
    if qa_pipeline is None:
        return "Sorry, detailed answers are unavailable due to a loading issue."
    context = f"{scheme.get('description', '')} Benefits: {scheme.get('benefits', '')} Eligibility: {scheme.get('eligibility', '')}"
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Step 5: Main Chatbot with Context and Advanced Features
def chatbot():
    print("🌟 Welcome to the Advanced Government Scheme Chatbot 3.0! 🌟")
    print("I can help with scheme info, eligibility, and details. Ask me anything or type 'exit' to quit.\n")

    schemes = load_schemes()
    if not schemes:
        print("🤖 Chatbot: No schemes loaded. Please fix 'scheme1.json'.")
        return

    context = {"last_category": None, "last_schemes": []}
    start_time = time.time()

    while True:
        user_input = input("💬 You: ")
        if user_input.lower() == "exit":
            print("🤖 Chatbot: Goodbye! See you next time! 👋")
            break

        query_start = time.time()
        lang = detect_language(user_input)
        entities = extract_entities(user_input)
        sentiment = analyze_sentiment(user_input)
        intent = recognize_intent(user_input, schemes)

        if intent == "scheme_info":
            matched_schemes, category = check_eligibility(user_input, entities, schemes, context)
            context["last_category"] = category
            context["last_schemes"] = matched_schemes

            if not matched_schemes or isinstance(matched_schemes[0], str):
                print(f"🤖 Chatbot: Oh no! 😔 I couldn’t find any schemes.")
                print(f"🤖 Chatbot: Try keywords like 'farmer', 'health', etc.")
            else:
                print(f"🤖 Chatbot: Found {len(matched_schemes)} schemes for {category}! Here they are:")
                for idx, scheme in enumerate(matched_schemes, 1):
                    scheme_name = scheme.get('name', 'N/A')
                    print(f"🤖 Chatbot: {idx}. *{scheme_name}*: {scheme.get('description', 'No description')}")
        elif intent == "eligibility" and context["last_schemes"]:
            for idx, scheme in enumerate(context["last_schemes"], 1):
                print(f"🤖 Chatbot: Eligibility for *{scheme.get('name', 'N/A')}*: {scheme.get('eligibility', 'Not specified')}")
        elif intent == "details" and context["last_schemes"]:
            if "which one" in user_input.lower() or "number" in user_input.lower():
                try:
                    idx = int(''.join(filter(str.isdigit, user_input))) - 1
                    scheme = context["last_schemes"][idx]
                    answer = answer_question(user_input, scheme)
                    print(f"🤖 Chatbot: For *{scheme.get('name', 'N/A')}*: {answer}")
                except (ValueError, IndexError):
                    print("🤖 Chatbot: Please specify a valid scheme number (e.g., 'details for number 1').")
            else:
                print("🤖 Chatbot: Which scheme? Say 'details for number X'.")
        else:
            print("🤖 Chatbot: Hmm, I’m not sure what you mean. Try asking about schemes, eligibility, or details!")

        query_end = time.time()
        print(f"🤖 Chatbot: ⏱️ Responded in {query_end - query_start:.2f} seconds.\n")

    total_time = time.time() - start_time
    print(f"🤖 Chatbot: 📊 Session Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    chatbot()