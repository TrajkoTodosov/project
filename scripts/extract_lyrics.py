import os
import whisper
from transformers import BertTokenizer
import json

# -------------------- CONFIG --------------------
RAW_AUDIO_DIR = os.path.join("data", "raw", "audio")
OUT_TEXT_DIR  = os.path.join("data", "processed", "lyrics_text")
OUT_TOKEN_DIR = os.path.join("data", "processed", "lyrics_tokens")
WHISPER_MODEL = "base"
TOKENIZER_MODEL = "bert-base-uncased"

os.makedirs(OUT_TEXT_DIR, exist_ok=True)
os.makedirs(OUT_TOKEN_DIR, exist_ok=True)

# load models
whisper_model = whisper.load_model(WHISPER_MODEL)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL)

# helper: transcribe and tokenize

def transcribe_and_tokenize(audio_path, out_text_dir, out_token_dir):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    # 1) whisper transcription
    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()
    # save plain text
    txt_path = os.path.join(out_text_dir, base + ".txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    # 2) tokenize with BERT
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors=None,
        truncation=True,
        max_length=512
    )
    # prepare a dict of input_ids and attention_mask
    token_data = {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask']
    }
    # save tokens as JSON
    tok_path = os.path.join(out_token_dir, base + '.json')
    with open(tok_path, 'w', encoding='utf-8') as f:
        json.dump(token_data, f)
    print(f"Processed lyrics for {base} (len={len(text)} chars, tokens={len(tokens['input_ids'])})")

# main
if __name__ == '__main__':
    files = [f for f in os.listdir(RAW_AUDIO_DIR) if f.lower().endswith(('.wav', '.mp3'))]
    for fname in files:
        path = os.path.join(RAW_AUDIO_DIR, fname)
        transcribe_and_tokenize(path, OUT_TEXT_DIR, OUT_TOKEN_DIR)