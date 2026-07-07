# =========================
# Flask App for Multi-Language Spelling Correction & Word Sense Disambiguation
# Supports: English, Hindi, Kannada
# =========================

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import json
import string
import os
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import assemblyai as aai
from difflib import SequenceMatcher

app = Flask(__name__)

# -------------------------
# Device Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# ⚠️ UPDATE THESE PATHS TO MATCH YOUR LOCAL FILE SYSTEM
# -------------------------
BASE_PATH = r"#YOUR_BASE_PATH"
DATASET_PATH = os.path.join(BASE_PATH, "dataset", "spelling dataset")

# Spelling Correction Configs
SPELLING_CONFIGS = {
    'english': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "best_seq2seq_attention.pt"),
        'vocab_path': None,
        'max_len': 256,
        'emb_dim': 128,
        'hid_dim': 256
    },
    'hindi': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "hindi_model", "best_model.pth"),
        'vocab_path': os.path.join(BASE_PATH, "models", "spelling checking models", "hindi_model", "vocab.json"),
        'max_len': 64,
        'emb_dim': 256,
        'hid_dim': 512
    },
    'kannada': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "kannada_model", "kannada_best_model.pth"),
        'vocab_path': os.path.join(BASE_PATH, "models", "spelling checking models", "kannada_model", "kannada_vocab.json"),
        'max_len': 64,
        'emb_dim': 256,
        'hid_dim': 512
    }
}

# WSD (Confusion) Configs
WSD_CONFIGS = {
    'english': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_wsd_model_improved.pt"),
        'model_name': 'bert-base-uncased',
        'tokenizer_type': 'bert'
    },
    'hindi': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_hindi_wsd_model.pt"),
        'model_name': 'google/muril-base-cased',
        'tokenizer_type': 'muril'
    },
    'kannada': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_kannada_wsd_model.pt"),
        'model_name': 'google/muril-base-cased',
        'tokenizer_type': 'muril'
    }
}

# ASSEMBLYAI_API_KEY Configs
ASSEMBLYAI_API_KEY = "#YOUR_ASSEMBLYAI_API_KEY"
aai.settings.api_key = ASSEMBLYAI_API_KEY


# -------------------------
# SPELLING CORRECTION: Model Architecture for ENGLISH
# -------------------------
class EncoderEnglish(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, pad_idx, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        hidden_reduced = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)
        cell_reduced = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
        return outputs, (hidden_reduced, cell_reduced)

class DecoderEnglish(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, enc_hid_dim, pad_idx, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, hid_dim, batch_first=True)
        self.attention = BahdanauAttention(enc_hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(hidden, encoder_outputs, mask=mask)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        pred_input = torch.cat((output, context, embedded), dim=1)
        prediction = self.fc_out(pred_input)
        return prediction, hidden, cell

# -------------------------
# SPELLING CORRECTION: Model Architecture for HINDI & KANNADA
# -------------------------
class EncoderIndic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2], cell[-1]), dim=1))).unsqueeze(0)
        return outputs, (hidden, cell)

class DecoderIndic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_hid, pad_idx, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hid, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim + enc_hid + emb_dim, vocab_size)
        self.attention = Attention(enc_hid, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

# -------------------------
# Shared Attention Modules
# -------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        dec_h = decoder_hidden.squeeze(0)
        src_len = encoder_outputs.size(1)
        dec_h_exp = dec_h.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_h_exp, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

# -------------------------
# Seq2Seq Wrapper (Spelling Correction)
# -------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def create_mask(self, src):
        return (src != self.pad_idx).to(self.device)

    def forward(self, src, max_len, vocab_size, sos_idx):
        batch_size = src.shape[0]
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        mask = self.create_mask(src)
        input_tok = torch.tensor([sos_idx] * batch_size).to(self.device)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell, encoder_outputs, mask=mask)
            outputs[:, t, :] = output
            input_tok = output.argmax(1)
        return outputs

# -------------------------
# WSD: English BERT Model
# -------------------------
class ImprovedBERTForWSD(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedBERTForWSD, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
       
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
       
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.layer_norm(pooled_output)
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
       
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
       
        return {'loss': loss, 'logits': logits}

# -------------------------
# WSD: Hindi/Kannada BERT Model
# -------------------------
class IndicBERTForWSD(nn.Module):
    def __init__(self, model_name, dropout_rate=0.3):
        super(IndicBERTForWSD, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
       
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
       
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.layer_norm(pooled_output)
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
       
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
       
        return {'loss': loss, 'logits': logits}

# -------------------------
# Vocab Loading (Spelling Correction)
# -------------------------
def build_english_vocab():
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    all_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    char_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(all_chars)
    char2idx = {ch: idx for idx, ch in enumerate(char_list)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return char2idx, idx2char

def load_vocab_from_json(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char2idx = vocab_data['char2idx']
    idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
    return char2idx, idx2char

# -------------------------
# Load Spelling Correction Vocabularies
# -------------------------
print("Loading spelling correction vocabularies...")
SPELLING_VOCABS = {'english': build_english_vocab()}
try:
    SPELLING_VOCABS['hindi'] = load_vocab_from_json(SPELLING_CONFIGS['hindi']['vocab_path'])
    SPELLING_VOCABS['kannada'] = load_vocab_from_json(SPELLING_CONFIGS['kannada']['vocab_path'])
    print("✅ All spelling vocabularies loaded successfully.")
except Exception as e:
    print("⚠️ Error loading spelling vocab:", e)

# -------------------------
# Load Spelling Correction Models
# -------------------------
SPELLING_MODELS = {}

def load_spelling_model(language):
    config = SPELLING_CONFIGS[language]
    char2idx, idx2char = SPELLING_VOCABS[language]
    vocab_size = len(char2idx)
    pad_idx = char2idx["<PAD>"]
   
    if language == 'english':
        dropout = 0.2
        enc = EncoderEnglish(vocab_size, config['emb_dim'], config['hid_dim'], pad_idx, dropout=dropout)
        dec = DecoderEnglish(vocab_size, config['emb_dim'], config['hid_dim'], config['hid_dim']*2, pad_idx, dropout=dropout)
    else:
        dropout = 0.3
        enc = EncoderIndic(vocab_size, config['emb_dim'], config['hid_dim'], pad_idx, dropout=dropout)
        dec = DecoderIndic(vocab_size, config['emb_dim'], config['hid_dim'], config['hid_dim']*2, pad_idx, dropout=dropout)
   
    model = Seq2Seq(enc, dec, device, pad_idx).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
   
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
   
    model.eval()
    return model

print("Loading spelling correction models...")
for lang in ['english', 'hindi', 'kannada']:
    try:
        SPELLING_MODELS[lang] = load_spelling_model(lang)
        print(f"✅ {lang.capitalize()} spelling model loaded.")
    except Exception as e:
        print(f"❌ Error loading {lang} spelling model:", e)

# -------------------------
# Load WSD Models and Tokenizers
# -------------------------
WSD_MODELS = {}
WSD_TOKENIZERS = {}

def load_wsd_model(language):
    config = WSD_CONFIGS[language]
   
    # Load tokenizer
    if config['tokenizer_type'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        model = ImprovedBERTForWSD(dropout_rate=0.4)
    else:  # muril for Hindi/Kannada
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        model = IndicBERTForWSD(config['model_name'], dropout_rate=0.4)
   
    model.to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
   
    return model, tokenizer

print("\nLoading WSD (confusion) models...")
for lang in ['english', 'hindi', 'kannada']:
    try:
        model, tokenizer = load_wsd_model(lang)
        WSD_MODELS[lang] = model
        WSD_TOKENIZERS[lang] = tokenizer
        print(f"✅ {lang.capitalize()} WSD model loaded.")
    except Exception as e:
        print(f"❌ Error loading {lang} WSD model:", e)

# -------------------------
# Spelling Correction Prediction Function
# -------------------------
def predict_spelling(text, language):
    if language not in SPELLING_MODELS:
        return {"error": f"Spelling model for '{language}' not loaded."}

    model = SPELLING_MODELS[language]
    char2idx, idx2char = SPELLING_VOCABS[language]
    config = SPELLING_CONFIGS[language]
    max_len = config['max_len']
    vocab_size = len(char2idx)

    sos_idx = char2idx["<SOS>"]
    eos_idx = char2idx["<EOS>"]
    pad_idx = char2idx["<PAD>"]
    unk_idx = char2idx["<UNK>"]

    enc_input = [sos_idx] + [char2idx.get(ch, unk_idx) for ch in list(text)[:max_len - 2]] + [eos_idx]
    enc_input += [pad_idx] * (max_len - len(enc_input))
    src = torch.tensor(enc_input).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(src, max_len, vocab_size, sos_idx)
        preds = output.argmax(dim=2).squeeze(0).cpu().numpy()

    corrected_chars = []
    for idx in preds:
        # Skip <SOS> and <PAD> tokens
        if idx == sos_idx or idx == pad_idx:
            continue
        # STOP when we hit <EOS> token
        if idx == eos_idx:
            break  # ← Changed from continue to break
        # Add valid character
        corrected_chars.append(idx2char.get(int(idx), ""))
   
    return {"original": text, "corrected": "".join(corrected_chars), "language": language}

# -------------------------
# WSD (Confusion Test) Prediction Function
# -------------------------
def predict_word_sense(context, gloss, target_word, language):
    if language not in WSD_MODELS:
        return {"error": f"WSD model for '{language}' not loaded."}
   
    model = WSD_MODELS[language]
    tokenizer = WSD_TOKENIZERS[language]
   
    model.eval()
   
    encoding = tokenizer.encode_plus(
        context,
        gloss,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
   
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
   
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
   
    sense_result = "Correct Sense" if prediction == 1 else "Wrong Sense"
   
    return {
        "context": context,
        "gloss": gloss,
        "target_word": target_word,
        "prediction": sense_result,
        "confidence": float(confidence),
        "language": language
    }

# -------------------------
# Flask Routes
# -------------------------
@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

# Login page
@app.route('/login')
def login():
    return render_template('authentication/login.html')

# Signup page
@app.route('/signup')
def signup():
    return render_template('authentication/signup.html')

@app.route('/correct_spelling', methods=['POST'])
def correct_spelling():
    """
    Spelling correction route - accepts language parameter
    Expected JSON: {"text": "...", "language": "english/hindi/kannada"}
    """
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'english').lower()
   
    if language not in ['english', 'hindi', 'kannada']:
        return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
           
    result = predict_spelling(text, language)
    return jsonify(result)

@app.route('/confusion_test', methods=['POST'])
def confusion_test():
    """
    Word Sense Disambiguation (Confusion Test) route - accepts language parameter
    Expected JSON: {
        "context": "...",
        "gloss": "...",
        "target_word": "...",
        "language": "english/hindi/kannada"
    }
    """
    data = request.get_json()
    context = data.get('context', '')
    gloss = data.get('gloss', '')
    target_word = data.get('target_word', '')
    language = data.get('language', 'english').lower()
   
    if not context or not gloss or not target_word:
        return jsonify({"error": "Missing required fields: context, gloss, or target_word"}), 400
   
    if language not in ['english', 'hindi', 'kannada']:
        return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
   
    result = predict_word_sense(context, gloss, target_word, language)
    return jsonify(result)

# Language codes for AssemblyAI
LANGUAGE_CODES = {
    'english': 'en',
    'hindi': 'hi',
    'kannada': 'kn'
}

# helper function for text similarity
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

# new route for speech test
@app.route('/speech_test', methods=['POST'])
def speech_test():
    """
    Speech test route using AssemblyAI API
    Expected form data:
    - audio: audio file (WAV format)
    - expected_text: the text user should speak
    - language: 'english', 'hindi', or 'kannada'
    """
    try:
        # Get form data
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
       
        audio_file = request.files['audio']
        expected_text = request.form.get('expected_text', '')
        language = request.form.get('language', 'english').lower()
       
        if not expected_text:
            return jsonify({"error": "Expected text is required"}), 400
       
        if language not in LANGUAGE_CODES:
            return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
       
        # Save audio file temporarily
        import tempfile
        import os
       
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        audio_file.save(temp_audio_path)
       
        try:
            # Configure AssemblyAI transcriber with language
            config = aai.TranscriptionConfig(
                language_code=LANGUAGE_CODES[language]
            )
           
            transcriber = aai.Transcriber(config=config)
           
            # Transcribe audio
            transcript = transcriber.transcribe(temp_audio_path)
           
            if transcript.status == aai.TranscriptStatus.error:
                return jsonify({"error": f"Transcription failed: {transcript.error}"}), 500
           
            transcribed_text = transcript.text if transcript.text else ""
           
            # Calculate similarity
            similarity = calculate_similarity(expected_text, transcribed_text)
           
            # Consider passed if similarity is above 70%
            is_passed = similarity >= 0.70
           
            return jsonify({
                "expected_text": expected_text,
                "transcribed_text": transcribed_text,
                "similarity": float(similarity),
                "is_passed": is_passed,
                "language": language
            })
           
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
   
    except Exception as e:
        print(f"Error in speech test: {str(e)}")
        return jsonify({"error": f"Speech processing failed: {str(e)}"}), 500

@app.route('/health')
def health():
    spelling_status = {lang: lang in SPELLING_MODELS for lang in ['english', 'hindi', 'kannada']}
    wsd_status = {lang: lang in WSD_MODELS for lang in ['english', 'hindi', 'kannada']}
   
    return jsonify({
        "status": "running",
        "device": str(device),
        "spelling_models": spelling_status,
        "wsd_models": wsd_status,
        "assemblyai": "configured" if ASSEMBLYAI_API_KEY else "not configured"
    })

# -------------------------
# Run App (LOCAL)
# -------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flask App Starting...")
    print("="*60)
    print("\n📍 Available Endpoints:")
    print("  • POST /correct_spelling - Spelling correction")
    print("  • POST /confusion_test - Word sense disambiguation")
    print("  • POST /speech_test - Speech test (AssemblyAI)")
    print("  • GET  /health - System status")
    print("\n" + "="*60)
    app.run(debug=True, port=5000)# =========================
# Flask App for Multi-Language Spelling Correction & Word Sense Disambiguation
# Supports: English, Hindi, Kannada
# =========================

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import json
import string
import os
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import assemblyai as aai
from difflib import SequenceMatcher

app = Flask(__name__)

# -------------------------
# Device Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# ⚠️ UPDATE THESE PATHS TO MATCH YOUR LOCAL FILE SYSTEM
# -------------------------
BASE_PATH = r"#YOUR_BASE_PATH"
DATASET_PATH = os.path.join(BASE_PATH, "dataset", "spelling dataset")

# Spelling Correction Configs
SPELLING_CONFIGS = {
    'english': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "best_seq2seq_attention.pt"),
        'vocab_path': None,
        'max_len': 256,
        'emb_dim': 128,
        'hid_dim': 256
    },
    'hindi': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "hindi_model", "best_model.pth"),
        'vocab_path': os.path.join(BASE_PATH, "models", "spelling checking models", "hindi_model", "vocab.json"),
        'max_len': 64,
        'emb_dim': 256,
        'hid_dim': 512
    },
    'kannada': {
        'model_path': os.path.join(BASE_PATH, "models", "spelling checking models", "kannada_model", "kannada_best_model.pth"),
        'vocab_path': os.path.join(BASE_PATH, "models", "spelling checking models", "kannada_model", "kannada_vocab.json"),
        'max_len': 64,
        'emb_dim': 256,
        'hid_dim': 512
    }
}

# WSD (Confusion) Configs
WSD_CONFIGS = {
    'english': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_wsd_model_improved.pt"),
        'model_name': 'bert-base-uncased',
        'tokenizer_type': 'bert'
    },
    'hindi': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_hindi_wsd_model.pt"),
        'model_name': 'google/muril-base-cased',
        'tokenizer_type': 'muril'
    },
    'kannada': {
        'model_path': os.path.join(BASE_PATH, "models", "confusion models",  "best_kannada_wsd_model.pt"),
        'model_name': 'google/muril-base-cased',
        'tokenizer_type': 'muril'
    }
}

# ASSEMBLYAI_API_KEY Configs
ASSEMBLYAI_API_KEY = "#YOUR_ASSEMBLYAI_API_KEY"
aai.settings.api_key = ASSEMBLYAI_API_KEY


# -------------------------
# SPELLING CORRECTION: Model Architecture for ENGLISH
# -------------------------
class EncoderEnglish(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, pad_idx, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        hidden_reduced = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)
        cell_reduced = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
        return outputs, (hidden_reduced, cell_reduced)

class DecoderEnglish(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, enc_hid_dim, pad_idx, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, hid_dim, batch_first=True)
        self.attention = BahdanauAttention(enc_hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(hidden, encoder_outputs, mask=mask)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        pred_input = torch.cat((output, context, embedded), dim=1)
        prediction = self.fc_out(pred_input)
        return prediction, hidden, cell

# -------------------------
# SPELLING CORRECTION: Model Architecture for HINDI & KANNADA
# -------------------------
class EncoderIndic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2], cell[-1]), dim=1))).unsqueeze(0)
        return outputs, (hidden, cell)

class DecoderIndic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_hid, pad_idx, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hid, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim + enc_hid + emb_dim, vocab_size)
        self.attention = Attention(enc_hid, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

# -------------------------
# Shared Attention Modules
# -------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        dec_h = decoder_hidden.squeeze(0)
        src_len = encoder_outputs.size(1)
        dec_h_exp = dec_h.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_h_exp, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

# -------------------------
# Seq2Seq Wrapper (Spelling Correction)
# -------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def create_mask(self, src):
        return (src != self.pad_idx).to(self.device)

    def forward(self, src, max_len, vocab_size, sos_idx):
        batch_size = src.shape[0]
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        mask = self.create_mask(src)
        input_tok = torch.tensor([sos_idx] * batch_size).to(self.device)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell, encoder_outputs, mask=mask)
            outputs[:, t, :] = output
            input_tok = output.argmax(1)
        return outputs

# -------------------------
# WSD: English BERT Model
# -------------------------
class ImprovedBERTForWSD(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedBERTForWSD, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
       
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
       
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.layer_norm(pooled_output)
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
       
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
       
        return {'loss': loss, 'logits': logits}

# -------------------------
# WSD: Hindi/Kannada BERT Model
# -------------------------
class IndicBERTForWSD(nn.Module):
    def __init__(self, model_name, dropout_rate=0.3):
        super(IndicBERTForWSD, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
       
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
       
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.layer_norm(pooled_output)
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
       
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
       
        return {'loss': loss, 'logits': logits}

# -------------------------
# Vocab Loading (Spelling Correction)
# -------------------------
def build_english_vocab():
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    all_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    char_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(all_chars)
    char2idx = {ch: idx for idx, ch in enumerate(char_list)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return char2idx, idx2char

def load_vocab_from_json(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char2idx = vocab_data['char2idx']
    idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
    return char2idx, idx2char

# -------------------------
# Load Spelling Correction Vocabularies
# -------------------------
print("Loading spelling correction vocabularies...")
SPELLING_VOCABS = {'english': build_english_vocab()}
try:
    SPELLING_VOCABS['hindi'] = load_vocab_from_json(SPELLING_CONFIGS['hindi']['vocab_path'])
    SPELLING_VOCABS['kannada'] = load_vocab_from_json(SPELLING_CONFIGS['kannada']['vocab_path'])
    print("✅ All spelling vocabularies loaded successfully.")
except Exception as e:
    print("⚠️ Error loading spelling vocab:", e)

# -------------------------
# Load Spelling Correction Models
# -------------------------
SPELLING_MODELS = {}

def load_spelling_model(language):
    config = SPELLING_CONFIGS[language]
    char2idx, idx2char = SPELLING_VOCABS[language]
    vocab_size = len(char2idx)
    pad_idx = char2idx["<PAD>"]
   
    if language == 'english':
        dropout = 0.2
        enc = EncoderEnglish(vocab_size, config['emb_dim'], config['hid_dim'], pad_idx, dropout=dropout)
        dec = DecoderEnglish(vocab_size, config['emb_dim'], config['hid_dim'], config['hid_dim']*2, pad_idx, dropout=dropout)
    else:
        dropout = 0.3
        enc = EncoderIndic(vocab_size, config['emb_dim'], config['hid_dim'], pad_idx, dropout=dropout)
        dec = DecoderIndic(vocab_size, config['emb_dim'], config['hid_dim'], config['hid_dim']*2, pad_idx, dropout=dropout)
   
    model = Seq2Seq(enc, dec, device, pad_idx).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
   
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
   
    model.eval()
    return model

print("Loading spelling correction models...")
for lang in ['english', 'hindi', 'kannada']:
    try:
        SPELLING_MODELS[lang] = load_spelling_model(lang)
        print(f"✅ {lang.capitalize()} spelling model loaded.")
    except Exception as e:
        print(f"❌ Error loading {lang} spelling model:", e)

# -------------------------
# Load WSD Models and Tokenizers
# -------------------------
WSD_MODELS = {}
WSD_TOKENIZERS = {}

def load_wsd_model(language):
    config = WSD_CONFIGS[language]
   
    # Load tokenizer
    if config['tokenizer_type'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        model = ImprovedBERTForWSD(dropout_rate=0.4)
    else:  # muril for Hindi/Kannada
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        model = IndicBERTForWSD(config['model_name'], dropout_rate=0.4)
   
    model.to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
   
    return model, tokenizer

print("\nLoading WSD (confusion) models...")
for lang in ['english', 'hindi', 'kannada']:
    try:
        model, tokenizer = load_wsd_model(lang)
        WSD_MODELS[lang] = model
        WSD_TOKENIZERS[lang] = tokenizer
        print(f"✅ {lang.capitalize()} WSD model loaded.")
    except Exception as e:
        print(f"❌ Error loading {lang} WSD model:", e)

# -------------------------
# Spelling Correction Prediction Function
# -------------------------
def predict_spelling(text, language):
    if language not in SPELLING_MODELS:
        return {"error": f"Spelling model for '{language}' not loaded."}

    model = SPELLING_MODELS[language]
    char2idx, idx2char = SPELLING_VOCABS[language]
    config = SPELLING_CONFIGS[language]
    max_len = config['max_len']
    vocab_size = len(char2idx)

    sos_idx = char2idx["<SOS>"]
    eos_idx = char2idx["<EOS>"]
    pad_idx = char2idx["<PAD>"]
    unk_idx = char2idx["<UNK>"]

    enc_input = [sos_idx] + [char2idx.get(ch, unk_idx) for ch in list(text)[:max_len - 2]] + [eos_idx]
    enc_input += [pad_idx] * (max_len - len(enc_input))
    src = torch.tensor(enc_input).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(src, max_len, vocab_size, sos_idx)
        preds = output.argmax(dim=2).squeeze(0).cpu().numpy()

    corrected_chars = []
    for idx in preds:
        # Skip <SOS> and <PAD> tokens
        if idx == sos_idx or idx == pad_idx:
            continue
        # STOP when we hit <EOS> token
        if idx == eos_idx:
            break  # ← Changed from continue to break
        # Add valid character
        corrected_chars.append(idx2char.get(int(idx), ""))
   
    return {"original": text, "corrected": "".join(corrected_chars), "language": language}

# -------------------------
# WSD (Confusion Test) Prediction Function
# -------------------------
def predict_word_sense(context, gloss, target_word, language):
    if language not in WSD_MODELS:
        return {"error": f"WSD model for '{language}' not loaded."}
   
    model = WSD_MODELS[language]
    tokenizer = WSD_TOKENIZERS[language]
   
    model.eval()
   
    encoding = tokenizer.encode_plus(
        context,
        gloss,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
   
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
   
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
   
    sense_result = "Correct Sense" if prediction == 1 else "Wrong Sense"
   
    return {
        "context": context,
        "gloss": gloss,
        "target_word": target_word,
        "prediction": sense_result,
        "confidence": float(confidence),
        "language": language
    }

# -------------------------
# Flask Routes
# -------------------------
@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

# Login page
@app.route('/login')
def login():
    return render_template('authentication/login.html')

# Signup page
@app.route('/signup')
def signup():
    return render_template('authentication/signup.html')

@app.route('/correct_spelling', methods=['POST'])
def correct_spelling():
    """
    Spelling correction route - accepts language parameter
    Expected JSON: {"text": "...", "language": "english/hindi/kannada"}
    """
    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'english').lower()
   
    if language not in ['english', 'hindi', 'kannada']:
        return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
           
    result = predict_spelling(text, language)
    return jsonify(result)

@app.route('/confusion_test', methods=['POST'])
def confusion_test():
    """
    Word Sense Disambiguation (Confusion Test) route - accepts language parameter
    Expected JSON: {
        "context": "...",
        "gloss": "...",
        "target_word": "...",
        "language": "english/hindi/kannada"
    }
    """
    data = request.get_json()
    context = data.get('context', '')
    gloss = data.get('gloss', '')
    target_word = data.get('target_word', '')
    language = data.get('language', 'english').lower()
   
    if not context or not gloss or not target_word:
        return jsonify({"error": "Missing required fields: context, gloss, or target_word"}), 400
   
    if language not in ['english', 'hindi', 'kannada']:
        return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
   
    result = predict_word_sense(context, gloss, target_word, language)
    return jsonify(result)

# Language codes for AssemblyAI
LANGUAGE_CODES = {
    'english': 'en',
    'hindi': 'hi',
    'kannada': 'kn'
}

# helper function for text similarity
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

# new route for speech test
@app.route('/speech_test', methods=['POST'])
def speech_test():
    """
    Speech test route using AssemblyAI API
    Expected form data:
    - audio: audio file (WAV format)
    - expected_text: the text user should speak
    - language: 'english', 'hindi', or 'kannada'
    """
    try:
        # Get form data
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
       
        audio_file = request.files['audio']
        expected_text = request.form.get('expected_text', '')
        language = request.form.get('language', 'english').lower()
       
        if not expected_text:
            return jsonify({"error": "Expected text is required"}), 400
       
        if language not in LANGUAGE_CODES:
            return jsonify({"error": "Language must be 'english', 'hindi', or 'kannada'"}), 400
       
        # Save audio file temporarily
        import tempfile
        import os
       
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        audio_file.save(temp_audio_path)
       
        try:
            # Configure AssemblyAI transcriber with language
            config = aai.TranscriptionConfig(
                language_code=LANGUAGE_CODES[language]
            )
           
            transcriber = aai.Transcriber(config=config)
           
            # Transcribe audio
            transcript = transcriber.transcribe(temp_audio_path)
           
            if transcript.status == aai.TranscriptStatus.error:
                return jsonify({"error": f"Transcription failed: {transcript.error}"}), 500
           
            transcribed_text = transcript.text if transcript.text else ""
           
            # Calculate similarity
            similarity = calculate_similarity(expected_text, transcribed_text)
           
            # Consider passed if similarity is above 70%
            is_passed = similarity >= 0.70
           
            return jsonify({
                "expected_text": expected_text,
                "transcribed_text": transcribed_text,
                "similarity": float(similarity),
                "is_passed": is_passed,
                "language": language
            })
           
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
   
    except Exception as e:
        print(f"Error in speech test: {str(e)}")
        return jsonify({"error": f"Speech processing failed: {str(e)}"}), 500

@app.route('/health')
def health():
    spelling_status = {lang: lang in SPELLING_MODELS for lang in ['english', 'hindi', 'kannada']}
    wsd_status = {lang: lang in WSD_MODELS for lang in ['english', 'hindi', 'kannada']}
   
    return jsonify({
        "status": "running",
        "device": str(device),
        "spelling_models": spelling_status,
        "wsd_models": wsd_status,
        "assemblyai": "configured" if ASSEMBLYAI_API_KEY else "not configured"
    })

# -------------------------
# Run App (LOCAL)
# -------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flask App Starting...")
    print("="*60)
    print("\n📍 Available Endpoints:")
    print("  • POST /correct_spelling - Spelling correction")
    print("  • POST /confusion_test - Word sense disambiguation")
    print("  • POST /speech_test - Speech test (AssemblyAI)")
    print("  • GET  /health - System status")
    print("\n" + "="*60)
    app.run(debug=True, port=5000)