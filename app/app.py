import streamlit as st
import json
import random
import pandas as pd
from datetime import datetime
import time
import streamlit.components.v1 as components
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. DIZIONARIO TRADUZIONI ---
# Qui definiamo tutte le parole che cambiano
T = {
    "title_demo": {
        "it": "📋 Studio di Valutazione Testi",
        "en": "📋 Text Evaluation Study"
    },
    "intro_demo": {
        "it": "Benvenuto. Prima di iniziare, inserisci alcune informazioni statistiche.",
        "en": "Welcome. Before starting, please provide some statistical info."
    },
    "age": {"it": "Età", "en": "Age"},
    "gender": {"it": "Genere", "en": "Gender"},
    "gender_opts": {
        "it": ["Uomo", "Donna", "Non binario", "Altro", "Preferisco non dire"],
        "en": ["Male", "Female", "Non-binary", "Other", "Prefer not to say"]
    },
    "edu": {"it": "Titolo di Studio", "en": "Education Level"},
    "edu_opts": {
        "it": ["Licenza Media", "Diploma", "Laurea Triennale", "Laurea Magistrale", "Dottorato"],
        "en": ["High School", "Bachelor's Degree", "Master's Degree", "PhD", "Other"]
    },
    "start_btn": {"it": "Inizia a Valutare", "en": "Start Evaluating"},
    "eval_title": {"it": "Valutazione", "en": "Evaluation"},
    "text_id_label": {"it": "Testo ID", "en": "Text ID"},
    "seen_label": {"it": "Visti", "en": "Seen"},
    "instructions": {
        "it": "Valuta il testo sopra secondo i seguenti criteri (1 = Minimo, 5 = Massimo):",
        "en": "Rate the text above according to the following criteria (1 = Lowest, 5 = Highest):"
    },
    "q1": {"it": "Quanto è CHIARO questo testo?", "en": "How CLEAR is this text?"},
    "q2": {"it": "Quanto è PERSUASIVO?", "en": "How PERSUASIVE is it?"},
    "q3": {"it": "Correttezza GRAMMATICALE?", "en": "GRAMMATICAL correctness?"},
    "submit_btn": {"it": "Invia Valutazione", "en": "Submit Evaluation"},
    "success_msg": {"it": "✅ Valutazione salvata! Caricamento prossimo testo...", "en": "✅ Saved! Loading next text..."},
    "finish_msg": {"it": "🎉 Hai valutato TUTTI i testi disponibili! Grazie.", "en": "🎉 You evaluated ALL available texts! Thank you."},
    "exit_btn": {"it": "Termina Sessione (Esci)", "en": "End Session (Exit)"},
    "thank_you": {"it": "Grazie per il tuo contributo!", "en": "Thank you for your contribution!"},
    "error_save": {"it": "Errore nel salvataggio", "en": "Error saving data"}
}

# --- FUNZIONI DI UTILITÀ ---
def scroll_to_top():
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTop = 0;
    </script>
    """
    components.html(js, height=0)

def get_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    # Assicurati che il nome sia corretto
    return client.open("texts_evaluation_sheet").sheet1 

@st.cache_data
def load_texts():
    # Carica il file JSON (assicurati che abbia il campo "lang")
    with open('dataset_small.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data_texts = load_texts()

# --- BLOCCO USCITA ---
if 'finito' in st.session_state and st.session_state['finito']:
    lang = st.session_state.get('language_choice', 'it') # Recupera lingua per saluto
    st.title(T['thank_you'][lang])
    st.balloons()
    st.stop()

# --- SELETTORE LINGUA (Sidebar o Top) ---
# Lo mettiamo nella sidebar o in alto. Qui lo metto in alto a destra usando le colonne.
col_logo, col_lang = st.columns([8, 2])
with col_lang:
    # Salviamo la scelta nello stato
    lang_code = st.radio("Language / Lingua", ["🇮🇹 Italiano", "🇬🇧 English"], index=0)
    
    # Convertiamo la scelta in 'it' o 'en' per usare il dizionario
    curr_lang = 'it' if "Italiano" in lang_code else 'en'
    st.session_state['language_choice'] = curr_lang

# --- INIZIALIZZAZIONE STATO ---
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None
if 'current_text' not in st.session_state:
    st.session_state['current_text'] = None
if 'seen_ids' not in st.session_state:
    st.session_state['seen_ids'] = []

# ==========================================
# FASE 1: DATI DEMOGRAFICI
# ==========================================
if st.session_state['user_info'] is None:
    st.title(T['title_demo'][curr_lang])
    st.write(T['intro_demo'][curr_lang])
    
    with st.form("demographics"):
        age = st.number_input(T['age'][curr_lang], min_value=18, max_value=99, step=1)
        
        # Le opzioni cambiano in base alla lingua
        gender = st.selectbox(T['gender'][curr_lang], T['gender_opts'][curr_lang])
        education = st.selectbox(T['edu'][curr_lang], T['edu_opts'][curr_lang])
        
        submit_demo = st.form_submit_button(T['start_btn'][curr_lang])
        
        if submit_demo:
            st.session_state['user_info'] = {
                "age": age,
                "gender": gender,
                "education": education,
                "language": curr_lang, # Salviamo anche la lingua scelta
                "session_id": str(datetime.now().timestamp())
            }
            st.rerun()

# ==========================================
# FASE 2: VALUTAZIONE
# ==========================================
else:
    placeholder_valutazione = st.empty()

    with placeholder_valutazione.container():
        st.title(T['eval_title'][curr_lang])
        
        # LOGICA DI PESCA (FILTRATA PER LINGUA)
        if st.session_state['current_text'] is None:
            # Filtro 1: Non ancora visti
            # Filtro 2: Corrispondono alla lingua scelta (campo 'lang' nel JSON)
            # Nota: useremo .get('lang', 'en') per defaultare a inglese se manca il campo
            testi_disponibili = [
                t for t in data_texts 
                if t['id'] not in st.session_state['seen_ids'] 
                and t.get('lang', 'en').lower() == curr_lang
            ]
            
            if not testi_disponibili:
                st.success(T['finish_msg'][curr_lang])
                st.session_state['finito'] = True
                if st.button("Termina"): # Bottone di fallback
                    st.stop()
                st.stop()
            
            st.session_state['current_text'] = random.choice(testi_disponibili)
        
        texto = st.session_state['current_text']
        
        # Mostra Testo
        label_id = T['text_id_label'][curr_lang]
        label_seen = T['seen_label'][curr_lang]
        st.info(f"📄 **{label_id}: {texto['id']}** ({label_seen}: {len(st.session_state['seen_ids'])})")
        
        st.markdown(f"### {texto['text']}")
        st.markdown("---")
        st.write(T['instructions'][curr_lang])
        
        # Form Valutazione
        with st.form("evaluation"):
            m1 = st.slider(T['q1'][curr_lang], 1, 5, 3)
            m2 = st.slider(T['q2'][curr_lang], 1, 5, 3)
            m3 = st.slider(T['q3'][curr_lang], 1, 5, 3)
            
            submit_eval = st.form_submit_button(T['submit_btn'][curr_lang])

    # LOGICA POST INVIO
    if submit_eval:
        placeholder_valutazione.empty()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = st.session_state['user_info']
        
        # Aggiungiamo user['language'] alla riga da salvare
        row_to_append = [
            timestamp,
            user['session_id'],
            user.get('language', 'it'), # Colonna lingua
            user['age'],
            user['gender'],
            user['education'],
            texto['id'],
            texto['text'],
            m1, m2, m3
        ]
        
        successo = False
        try:
            sheet = get_google_sheet()
            sheet.append_row(row_to_append)
            successo = True
        except Exception as e:
            if "200" in str(e):
                successo = True
            else:
                st.error(f"{T['error_save'][curr_lang]}: {e}")

        if successo:
            st.session_state['seen_ids'].append(texto['id'])
            st.success(T['success_msg'][curr_lang])
            
            st.session_state['current_text'] = None
            scroll_to_top()
            time.sleep(1.5)
            st.rerun()

    if not submit_eval:
        st.markdown("---")
        if st.button(T['exit_btn'][curr_lang]):
            st.session_state['finito'] = True
            st.rerun()