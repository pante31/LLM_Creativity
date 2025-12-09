import streamlit as st
import json
import random
import pandas as pd
from datetime import datetime

# 1. Caricamento dei testi (cache per non ricaricare ogni volta)
@st.cache_data
def load_data():
    with open('testi.json', 'r') as f:
        data = json.load(f)
    return data

data = load_data()

# 2. Gestione dello Stato (Memoria dell'utente)
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None # Dati demografici non ancora inseriti
if 'current_text' not in st.session_state:
    st.session_state['current_text'] = None # Nessun testo selezionato al momento

# --- FASE 1: RACCOLTA DEMOGRAFICA ---
if st.session_state['user_info'] is None:
    st.title("Benvenuto allo studio di valutazione")
    st.write("Per iniziare, inserisci alcune informazioni su di te.")
    
    with st.form("demographics_form"):
        age = st.number_input("Età", min_value=18, max_value=100)
        gender = st.selectbox("Genere", ["Uomo", "Donna", "Non binario", "Preferisco non specificare"])
        education = st.selectbox("Titolo di studio", ["Diploma", "Laurea Triennale", "Laurea Magistrale", "Dottorato"])
        
        submitted = st.form_submit_button("Inizia Valutazione")
        
        if submitted:
            # Salviamo i dati nella sessione e ricarichiamo la pagina
            st.session_state['user_info'] = {
                "age": age,
                "gender": gender,
                "education": education,
                "session_id": str(datetime.now().timestamp()) # ID unico per l'utente
            }
            st.rerun()

# --- FASE 2: VALUTAZIONE TESTI ---
else:
    st.title("Valuta il testo")
    
    # Se non c'è un testo attivo, ne scegliamo uno a caso
    if st.session_state['current_text'] is None:
        st.session_state['current_text'] = random.choice(data)
    
    text_to_show = st.session_state['current_text']
    
    st.markdown(f"### Leggi il seguente testo:\n\n> {text_to_show['text']}")
    st.markdown("---")
    
    # Form di valutazione
    with st.form("evaluation_form"):
        metric_1 = st.slider("Quanto è chiaro questo testo?", 1, 5)
        metric_2 = st.slider("Quanto è persuasivo?", 1, 5)
        metric_3 = st.radio("Grammatica corretta?", [1, 2, 3, 4, 5], horizontal=True)
        
        submit_eval = st.form_submit_button("Invia Valutazione")
        
        if submit_eval:
            # Qui raccogliamo TUTTI i dati (Demografici + Valutazione + ID Testo)
            result = {
                **st.session_state['user_info'], # Espande i dati demografici
                "text_id": text_to_show['id'],
                "clarity": metric_1,
                "persuasiveness": metric_2,
                "grammar": metric_3,
                "timestamp": datetime.now()
            }
            
            # --- SALVATAGGIO (Vedi Step 3) ---
            save_to_database(result) 
            
            st.success("Valutazione salvata!")
            
            # Reset del testo per pescarne uno nuovo al prossimo giro
            st.session_state['current_text'] = None
            
            # Bottoni per continuare o uscire
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Valuta un altro testo"):
                    st.rerun()
            with col2:
                if st.button("Termina sessione"):
                    st.stop()