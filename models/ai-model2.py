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

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

from transformers import BertModel


model_path = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
indic_bert = BertModel.from_pretrained(model_path)