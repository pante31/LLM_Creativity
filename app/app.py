import streamlit as st
import json
import random
import os
import pandas as pd
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAZIONE GOOGLE SHEETS ---
# Questa funzione si collega al foglio usando le "chiavi segrete"
def get_google_sheet():
    # Definiamo i permessi necessari
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    # Carichiamo le credenziali dai "Secrets" di Streamlit (vedi Fase 3)
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    
    client = gspread.authorize(creds)
    
    # QUI devi mettere il nome esatto del tuo foglio Google
    sheet = client.open("Valutazione_Testi_Data").sheet1 
    return sheet

# --- CARICAMENTO DATI ---
@st.cache_data
def load_texts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "dataset_small.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

data_texts = load_texts()

# --- GESTIONE STATO UTENTE ---
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None
if 'current_text' not in st.session_state:
    st.session_state['current_text'] = None

# --- PARTE 1: QUESTIONARIO DEMOGRAFICO ---
if st.session_state['user_info'] is None:
    st.title("📋 Studio di Valutazione Testi")
    st.markdown("Benvenuto. Prima di iniziare, ti chiediamo alcune informazioni anonime per fini statistici.")
    
    with st.form("demographics"):
        age = st.number_input("La tua età", min_value=18, max_value=99, step=1)
        gender = st.selectbox("Genere", ["Uomo", "Donna", "Non binario", "Altro", "Preferisco non specificare"])
        education = st.selectbox("Titolo di Studio", ["Licenza Media", "Diploma", "Laurea Triennale", "Laurea Magistrale", "Dottorato/Post-Laurea"])
        
        submit_demo = st.form_submit_button("Inizia a Valutare")
        
        if submit_demo:
            # Salviamo i dati in memoria
            st.session_state['user_info'] = {
                "age": age,
                "gender": gender,
                "education": education,
                "session_id": str(datetime.now().timestamp()) # ID unico per sessione
            }
            st.rerun()

# --- PARTE 2: VALUTAZIONE TESTI ---
else:
    st.title("Valutazione")
    
    # Se non c'è un testo selezionato, ne prendiamo uno a caso
    if st.session_state['current_text'] is None:
        st.session_state['current_text'] = random.choice(data_texts)
    
    texto = st.session_state['current_text']
    
    # Mostriamo il testo in un box evidente
    st.info(f"📄 **Prompt Testo {texto['prompt']}**")
    st.markdown(f"### {texto['text']}")
    st.markdown("---")
    
    st.write("Valuta il testo sopra secondo i seguenti criteri (1 = Minimo, 5 = Massimo):")
    
    with st.form("evaluation"):
        m1 = st.slider("Quanto è CHIARO questo testo?", 1, 5, 3)
        m2 = st.slider("Quanto è PERSUASIVO?", 1, 5, 3)
        m3 = st.slider("Correttezza GRAMMATICALE?", 1, 5, 3)
        
        submit_eval = st.form_submit_button("Invia Valutazione")
        
        if submit_eval:
            # 1. Prepariamo la riga da salvare
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user = st.session_state['user_info']
            
            row_to_append = [
                timestamp,
                user['session_id'],
                user['age'],
                user['gender'],
                user['education'],
                texto['id'],
                texto['text'], # Salviamo anche il testo per comodità di lettura nel foglio
                m1,
                m2,
                m3
            ]
            
            # 2. Inviamo a Google Sheets
            try:
                sheet = get_google_sheet()
                sheet.append_row(row_to_append)
                st.success("✅ Valutazione salvata con successo!")
            except Exception as e:
                st.error(f"Errore nel salvataggio: {e}")
            
            # 3. Reset del testo per il prossimo giro
            st.session_state['current_text'] = None
            
            # Opzioni per continuare
            col1, col2 = st.columns(2)
            with col1:
                st.write("Vuoi valutare un altro testo?")
                st.button("Sì, continua", on_click=st.rerun) # Ricarica e pesca nuovo testo
            with col2:
                if st.button("No, ho finito"):
                    st.stop() # Ferma l'app