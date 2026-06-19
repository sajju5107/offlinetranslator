import streamlit as st
import sqlite3
import pyttsx3
import os
import torch
import pdfplumber
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
from deep_translator import GoogleTranslator

# --- Windows Tesseract Configuration ---
# If running on Windows, uncomment the line below and map it to your local Tesseract install folder:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ----------------- 1. DATABASE & ENGINES -----------------
def init_local_db():
    conn = sqlite3.connect('global_translator.db')
    conn.execute('CREATE TABLE IF NOT EXISTS logs (user TEXT, src_txt TEXT, res_txt TEXT, pair TEXT, mode TEXT, time TEXT)')
    conn.commit()
    conn.close()

# Feature: History automatically erased on refresh
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    conn = sqlite3.connect('global_translator.db')
    # FIX: Create the table first so the database doesn't crash when trying to clear it
    conn.execute('CREATE TABLE IF NOT EXISTS logs (user TEXT, src_txt TEXT, res_txt TEXT, pair TEXT, mode TEXT, time TEXT)')
    conn.execute("DELETE FROM logs")  # Now it is safe to wipe history
    conn.commit()
    conn.close()

@st.cache_resource
def load_offline_engines():
    with st.status("Initializing Global AI Engines (Offline Mode)...", expanded=False):
        stt = WhisperModel("base", device="cpu", compute_type="int8") 
        model_id = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return stt, tokenizer, model

# Helper to map NLLB codes to Google/gTTS 2-letter codes safely
def get_online_lang_code(nllb_code):
    nllb_to_online_exceptions = {
        "kan": "kn", "ben": "bn", "guj": "gu", "mal": "ml", 
        "mar": "mr", "pan": "pa", "urd": "ur", "hin": "hi", 
        "tam": "ta", "tel": "te", "ory": "or"
    }
    prefix = nllb_code.split('_')[0]
    return nllb_to_online_exceptions.get(prefix, prefix[:2])

# ----------------- 2. UNIVERSAL LOGIC -----------------
def perform_translation(text, src_code, tgt_code, mode, nllb_tok=None, nllb_model=None):
    if not text.strip(): return ""
    if mode == "Online":
        online_code = get_online_lang_code(tgt_code)
        return GoogleTranslator(source='auto', target=online_code).translate(text)
    else:
        nllb_tok.src_lang = src_code
        inputs = nllb_tok(text, return_tensors="pt")
        tgt_id = nllb_tok.convert_tokens_to_ids(tgt_code)
        tokens = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=250)
        return nllb_tok.batch_decode(tokens, skip_special_tokens=True)[0]

def perform_tts(text, lang_code, mode):
    if not text.strip(): return
    if mode == "Online":
        online_code = get_online_lang_code(lang_code)
        tts = gTTS(text=text, lang=online_code)
        tts.save("temp_speech.mp3")
        st.audio("temp_speech.mp3", format="audio/mp3", autoplay=True)
    else:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

def log_to_history(text, translated, src_name, tgt_name, mode):
    conn = sqlite3.connect('global_translator.db')
    conn.execute(
        "INSERT INTO logs (user, src_txt, res_txt, pair, mode, time) VALUES (?, ?, ?, ?, ?, ?)", 
        ("User", text, translated, f"{src_name}->{tgt_name}", mode, datetime.now())
    )
    conn.commit()
    conn.close()

# ----------------- 3. UI & LANGUAGE DICTIONARY -----------------
st.set_page_config(page_title="Global Hybrid AI Translator", layout="wide")
init_local_db()

ALL_LANGS = {
    # All Core Indian Languages
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Tamil": "tam_Taml", "Telugu": "tel_Telu", 
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru", "Urdu": "urd_Arab", "Assamese": "asm_Beng", "Odia": "ory_Orya",
    # Global Languages
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", "German": "deu_Latn",
    "Chinese (Simplified)": "zho_Hans", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Russian": "rus_Cyrl", "Arabic": "arb_Arab", "Portuguese": "por_Latn", "Italian": "ita_Latn"
}

if "manual_out" not in st.session_state: st.session_state.manual_out = ""
if "file_out" not in st.session_state: st.session_state.file_out = ""

# Sidebar Settings
with st.sidebar:
    st.title("Settings⚙️")
    current_mode = "Online" if st.toggle("🌐 Online Mode", value=False) else "Offline"
    st.info(f"Active Mode: **{current_mode}**")
    st.divider()
    
    st.write("### Session History 🕒")
    conn = sqlite3.connect('global_translator.db')
    history = conn.execute("SELECT res_txt, pair FROM logs ORDER BY time DESC LIMIT 5").fetchall()
    conn.close()
    
    if history:
        for res, pair in history: 
            st.caption(f"**[{pair}]**: {res[:50]}...")
    else:
        st.caption("No history items in this session. (Erases on refresh)")

# Load Offline Engines if Needed
n_tok, n_mod, s_eng = None, None, None
if current_mode == "Offline":
    s_eng, n_tok, n_mod = load_offline_engines()

# Header Layout
st.title("🌎 Multi-Feature Hybrid AI Translator")
col_l, col_r = st.columns(2)
with col_l: src_l = st.selectbox("From Language", list(ALL_LANGS.keys()), index=12)  # Default: English
with col_r: tgt_l = st.selectbox("To Language", list(ALL_LANGS.keys()), index=6)    # Default: Kannada

st.divider()

# Core Navigation Tabs
tab_text, tab_voice, tab_files = st.tabs(["📝 Text Entry", "🎙️ Speech-to-Text Voice", "📄 Documents & OCR"])

# --- TAB 1: MANUAL TEXT INPUT ---
with tab_text:
    t1, t2 = st.columns(2)
    with t1:
        manual_in = st.text_area("Type text manually:", height=180)
        if st.button("Translate Text ➔", key="text_btn"):
            st.session_state.manual_out = perform_translation(manual_in, ALL_LANGS[src_l], ALL_LANGS[tgt_l], current_mode, n_tok, n_mod)
            log_to_history(manual_in, st.session_state.manual_out, src_l, tgt_l, current_mode)
    with t2:
        st.text_area("Translation Result:", value=st.session_state.manual_out, height=180, disabled=True)
        if st.button("🔊 Read Translation Output", key="tts_text_btn"):
            perform_tts(st.session_state.manual_out, ALL_LANGS[tgt_l], current_mode)

# --- TAB 2: SPEECH-TO-TEXT / LIVE VOICE ---
with tab_voice:
    st.subheader("Live Audio Conversation Translation")
    st.write("Record voice snippet. The engine will decode, translate, and synthesize speech output.")
    
    def handle_voice(audio_data, s_name, t_name):
        with open("live.wav", "wb") as f: 
            f.write(audio_data['bytes'])
        
        if current_mode == "Online":
            r = sr.Recognizer()
            with sr.AudioFile("live.wav") as source:
                try: text = r.recognize_google(r.record(source))
                except: text = ""
        else:
            segments, _ = s_eng.transcribe("live.wav")
            text = " ".join([s.text for s in segments])

        if text.strip():
            translated = perform_translation(text, ALL_LANGS[s_name], ALL_LANGS[t_name], current_mode, n_tok, n_mod)
            st.chat_message("user").write(f"**{s_name} (Spoken):** {text}")
            st.chat_message("assistant").write(f"**{t_name} (Translated):** {translated}")
            
            log_to_history(text, translated, s_name, t_name, current_mode)
            perform_tts(translated, ALL_LANGS[t_name], current_mode)

    v1, v2 = st.columns(2)
    with v1:
        st.write(f"Record **{src_l}** Input Mic:")
        audio_1 = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop & Process", key='m1')
        if audio_1: handle_voice(audio_1, src_l, tgt_l)
    with v2:
        st.write(f"Record **{tgt_l}** Input Mic:")
        audio_2 = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop & Process", key='m2')
        if audio_2: handle_voice(audio_2, tgt_l, src_l)

# --- TAB 3: FILE TRANSLATION (PDF, IMAGE OCR, CAMERA OCR) ---
with tab_files:
    st.subheader("Process External Documents & Document Graphics")
    
    file_mode = st.radio("Choose Input Medium:", ["File Upload (PDF / Image)", "Live Camera Stream Capture"])
    extracted_text = ""

    if file_mode == "File Upload (PDF / Image)":
        uploaded_file = st.file_uploader("Upload File Document (.pdf, .png, .jpg, .jpeg)", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Feature: PDF Translation
            if file_ext == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    pages_txt = [page.extract_text() for page in pdf.pages if page.extract_text()]
                    extracted_text = "\n".join(pages_txt)
            
            # Feature: Image OCR Translation
            elif file_ext in ["png", "jpg", "jpeg"]:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Document Graphic Target", width=350)
                extracted_text = pytesseract.image_to_string(img)

    # Feature: Camera OCR Translation
    else:
        cam_shot = st.camera_input("Position visual document in front of camera lens and capture:")
        if cam_shot is not None:
            img = Image.open(cam_shot)
            extracted_text = pytesseract.image_to_string(img)

    # Process block for parsed or captured content
    if extracted_text.strip():
        st.success("Successfully Extracted Raw Text Content!")
        f1, f2 = st.columns(2)
        with f1:
            st.text_area("Extracted Input Text:", value=extracted_text, height=200, disabled=True)
            if st.button("Translate Extracted Content ➔", key="file_process_btn"):
                st.session_state.file_out = perform_translation(extracted_text, ALL_LANGS[src_l], ALL_LANGS[tgt_l], current_mode, n_tok, n_mod)
                log_to_history(extracted_text[:100], st.session_state.file_out, src_l, tgt_l, current_mode)
        with f2:
            st.text_area("Translated Document Result:", value=st.session_state.file_out, height=200, disabled=True)
            if st.button("🔊 Read Aloud File Translation Output", key="tts_file_btn"):
                perform_tts(st.session_state.file_out, ALL_LANGS[tgt_l], current_mode)
