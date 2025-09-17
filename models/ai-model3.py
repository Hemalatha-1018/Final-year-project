# Step 1: Load the dataset
def load_schemes():
    with open('/work/data/myscheme/scheme1.json', 'r') as file:
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
        elif ent.label_ == "ORG" or ent.label_ == "PERSON":
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


# Step 3: Seq2Seq Model with Attention Mechanism (Simplified)
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
        x = trg[:, 0:1, :]  # Start token
        for t in range(1, trg.shape[1]):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs, self.attention)
            outputs.append(output)
            x = output
        return torch.cat(outputs, dim=1)