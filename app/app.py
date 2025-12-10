import json
import random
import os
import time
import gspread
import streamlit as st
import pandas as pd

from datetime import datetime
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
    sheet = client.open("texts_evaluation_sheet").sheet1 
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
    
    # Mostriamo il testo
    st.info(f"📄 **Testo ID: {texto['id']}**")
    st.markdown(f"### {texto['text']}")
    st.markdown("---")
    
    st.write("Valuta il testo sopra secondo i seguenti criteri (1 = Minimo, 5 = Massimo):")
    
    # --- FORM DI VALUTAZIONE ---
    with st.form("evaluation"):
        m1 = st.slider("Quanto è CHIARO questo testo?", 1, 5, 3)
        m2 = st.slider("Quanto è PERSUASIVO?", 1, 5, 3)
        m3 = st.slider("Correttezza GRAMMATICALE?", 1, 5, 3)
        
        # Unico bottone permesso dentro il form
        submit_eval = st.form_submit_button("Invia Valutazione")
        
    # --- LOGICA POST INVIO (Fuori dal form) ---
    if submit_eval:
        # 1. Prepariamo la riga
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = st.session_state['user_info']
        
        row_to_append = [
            timestamp,
            user['session_id'],
            user['age'],
            user['gender'],
            user['education'],
            texto['id'],
            texto['text'],
            m1,
            m2,
            m3
        ]
        
        # Variabile per tracciare se è andato tutto bene
        successo = False
        
        # 2. Inviamo a Google Sheets
        try:
            sheet = get_google_sheet()
            sheet.append_row(row_to_append)
            successo = True # Se arriva qui, ha funzionato
            
        except Exception as e:
            # TRUCCO: Se l'errore contiene "200", in realtà è un successo!
            if "200" in str(e):
                successo = True
            else:
                st.error(f"Errore tecnico: {e}")

        # 3. SE È ANDATO TUTTO BENE (O SEMBRAVA UN ERRORE 200)
        if successo:
            st.success("✅ Valutazione salvata! Caricamento prossimo testo...")
            
            # Reset del testo
            st.session_state['current_text'] = None
            
            # Pausa per leggere il messaggio
            time.sleep(1.5) 
            
            # Ricarica la pagina (FUORI dal try/except per evitare errori)
            st.rerun()
    
    # Tasto per uscire
    st.markdown("---")
    if st.button("Termina Sessione (Esci)"):
        st.session_state.clear()
        st.success("Grazie per il tuo contributo!")
        time.sleep(1)
        st.stop()