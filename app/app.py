import json
import random
import os
import time
import gspread
import streamlit as st
import pandas as pd


from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# Se l'utente ha premuto "Esci", mostriamo i saluti e blocchiamo tutto QUI.
if 'finito' in st.session_state and st.session_state['finito']:
    st.title("Grazie!")
    st.success("La tua sessione è terminata. Puoi chiudere questa scheda.")
    st.balloons() # Un po' di festa opzionale
    st.stop() # QUI st.stop() funziona perché impedisce di caricare tutto il resto sotto!

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
    # 1. Creiamo un contenitore vuoto che conterrà TUTTA l'interfaccia di valutazione
    placeholder_valutazione = st.empty()

    # 2. Costruiamo l'interfaccia DENTRO questo contenitore
    with placeholder_valutazione.container():
        st.title("Valutazione")
        
        # Logica di pesca del testo
        if st.session_state['current_text'] is None:
            st.session_state['current_text'] = random.choice(data_texts)
        
        texto = st.session_state['current_text']
        
        # Mostriamo il testo
        st.info(f"📄 **Testo ID: {texto['id']}**")
        st.markdown(f"### {texto['text']}")
        st.markdown("---")
        st.write("Valuta il testo sopra secondo i seguenti criteri (1 = Minimo, 5 = Massimo):")
        
        # Form
        with st.form("evaluation"):
            m1 = st.slider("Quanto è CHIARO questo testo?", 1, 5, 3)
            m2 = st.slider("Quanto è PERSUASIVO?", 1, 5, 3)
            m3 = st.slider("Correttezza GRAMMATICALE?", 1, 5, 3)
            submit_eval = st.form_submit_button("Invia Valutazione")

    # --- LOGICA POST INVIO (Fuori dal contenitore visivo) ---
    if submit_eval:
        # 3. TRUCCO MAGICO: Svuotiamo immediatamente l'interfaccia!
        # La pagina ora è vuota, quindi il browser torna in cima per forza.
        placeholder_valutazione.empty()
        
        # Salvataggio dati
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = st.session_state['user_info']
        
        row_to_append = [
            timestamp, user['session_id'], user['age'], user['gender'], 
            user['education'], texto['id'], texto['text'], m1, m2, m3
        ]
        
        successo = False
        try:
            sheet = get_google_sheet()
            sheet.append_row(row_to_append)
            successo = True
        except Exception as e:
            if "200" in str(e): # Gestione falso positivo gspread
                successo = True
            else:
                st.error(f"Errore tecnico: {e}")

        if successo:
            # Ora mostriamo il messaggio di successo (apparirà in alto perché il resto è sparito)
            st.success("✅ Valutazione salvata! Caricamento prossimo testo...")
            
            st.session_state['current_text'] = None
            time.sleep(5)
            st.rerun()

    # Tasto per uscire (lo mettiamo fuori dal placeholder così sparisce o resta a seconda della logica)
    if not submit_eval: # Lo mostriamo solo se non stiamo salvando
        st.markdown("---")
        if st.button("Termina Sessione (Esci)"):
            st.session_state['finito'] = True
            st.rerun()