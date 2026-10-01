# ==============================================================================
# PROJET DE FIN D'ÉTUDES : RAG INDUSTRIEL POUR VARIATEURS ABB ACS880
# ------------------------------------------------------------------------------
# Auteur          : Ameur Bacem
# Établissement   : École Nationale d'Ingénieurs de Monastir (ENIM)
# Département     : Génie Électrique (3ème Année)
# Année Académique: 2025 - 2026
# Licence         : Propriété Intellectuelle Exclusive - Reproduction Interdite
# ==============================================================================
__author__ = "Ameur Bacem"
__copyright__ = "Copyright 2026, Ameur Bacem - ENIM"
__credits__ = ["Ameur Bacem"]
__version__ = "2.1.0"
__maintainer__ = "Ameur Bacem"

import os
import re
import time
import json
import sqlite3
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import edge_tts

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
# ==============================================================================
# CONFIGURATION STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="ABB ACS880 - Cockpit Diagnostic",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #FF000F; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .incident-card {
        background-color: #f8f9fa;
        border-left: 4px solid #FF000F;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.95rem;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# GESTIONNAIRE DE BASE SQLITE (MÉMOIRE 4 HEURES)
# ==============================================================================
DB_FILE = "maintenance_memory.db"
SESSION_TTL_SECONDS = 4 * 3600  # 4 heures

def init_memory_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            updated_at REAL,
            messages_json TEXT,
            incident_summary TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_session_to_disk(session_id, messages, summary):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO sessions (session_id, updated_at, messages_json, incident_summary)
        VALUES (?, ?, ?, ?)
    """, (session_id, time.time(), json.dumps(messages), summary))
    conn.commit()
    conn.close()

def load_session_from_disk(session_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT updated_at, messages_json, incident_summary FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        updated_at, messages_json, summary = row
        if (time.time() - updated_at) <= SESSION_TTL_SECONDS:
            return json.loads(messages_json), summary, updated_at
        else:
            delete_session_from_disk(session_id)
            return [], "Session expirée après 4 heures d'inactivité.", None
    return [], "En attente de signalement de panne...", None

def delete_session_from_disk(session_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

init_memory_db()

# ==============================================================================
# MOTEUR RAG EN CACHE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    DB_DIR = "chroma_db_abb_multi"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    groq_client = Groq()
    return vector_db, llm, groq_client

with st.spinner("⚡ Connexion au moteur d'IA ABB ACS880 (Modèle 120B)..."):
    vector_db, llm, groq_client = load_rag_engine()

# ==============================================================================
# PARSEUR ROBUSTE DES RÉPONSES DE L'IA
# ==============================================================================
def parse_llm_output(raw_text):
    """Extrait proprement le résumé d'incident, le vocal et le message public sans balises parasites."""
    summary_match = re.search(r'\[SYNTHESE_INCIDENT\](.*?)(\[/SYNTHESE_INCIDENT\]|$)', raw_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""

    vocal_match = re.search(r'\[VOCAL\](.*?)(\[/VOCAL\]|$)', raw_text, re.DOTALL)
    vocal = vocal_match.group(1).strip() if vocal_match else ""

    # Nettoyage du message affiché au technicien
    clean_text = raw_text
    clean_text = re.sub(r'\[SYNTHESE_INCIDENT\].*?(\[/SYNTHESE_INCIDENT\]|\n)', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\[VOCAL\].*?(\[/VOCAL\]|\n)', '', clean_text, flags=re.DOTALL)
    clean_text = clean_text.replace('[REPONSE]', '').replace('[/REPONSE]', '').strip()

    return summary, vocal, clean_text

# ==============================================================================
# PROMPTS AGENTIQUES
# ==============================================================================
query_rewrite_prompt = PromptTemplate(
    template="""You are an expert technical search agent for ABB ACS880 drives.
Analyze the incident history and the technician's latest response (which could simply be 'Oui', 'Non', or a brief test result).
Identify the physical fault being diagnosed (e.g. fault 3220 undervoltage, DC bus, main contactor, input supply phase).
Formulate a concise, highly specific English search query for the ABB Hardware and Firmware manuals.

HISTORIQUE DE L'INCIDENT :
{summary}

DERNIÈRE ENTRÉE DU TECHNICIEN :
{question}

ENGLISH SEARCH QUERY (Keywords only):""",
    input_variables=["summary", "question"]
)

diagnostic_prompt = PromptTemplate(
    template="""Tu es un ingénieur expert senior en maintenance des variateurs industriels ABB ACS880.
Tu guides un technicien en intervention sur site. Tu as accès aux manuels techniques d'ABB.

CARNET DE BORD D'INCIDENT ACTUEL :
{summary}

HISTORIQUE RÉCENT DES MESSAGES :
{history}

EXTRAITS DU MANUEL ABB :
{context}

DERNIER MESSAGE DU TECHNICIEN :
{question}

PROTOCOLE DE COMPORTEMENT :
CAS 1 : SI LE PROBLÈME MANQUE DE DÉTAILS OU RESTE AMBIGU :
- Propose 2 ou 3 questions simples de vérification terrain auxquelles le technicien peut répondre par OUI ou par NON.
- Exemple :
  1. Le code défaut 3220 (ou similaire) apparaît-il sur l'écran du panneau ? (OUI/NON)
  2. La coupure se produit-elle dès la mise sous tension avant même la commande RUN ? (OUI/NON)
  3. Avez-vous mesuré une tension triphasée ≥ 380V entre L1, L2 et L3 ? (OUI/NON)
- Conclus en disant qu'il peut aussi répondre librement s'il a d'autres éléments.

CAS 2 : SI LE TECHNICIEN A RÉPONDU AUX QUESTIONS (ex: "Oui à la 1 et la 2") OU SI LE CODE EST IDENTIFIÉ :
- Rédige la directive technique complète d'intervention :
  - ⚡ Consignation, VAT et règles de sécurité.
  - 🔍 Analyse de la cause racine exacte.
  - 🛠️ Tableau d'intervention avec Bornes, Actions, Couples de serrage en N·m, et Paramètres firmware à vérifier.
  - 📖 Pages et manuels cités.

STRUCTURE DU FORMAT :
[SYNTHESE_INCIDENT]
Synthèse de l'état actuel pour le dossier en 2 ou 3 phrases.
[/SYNTHESE_INCIDENT]

[VOCAL]
2 ou 3 phrases courtes et percutantes dites au technicien.
[/VOCAL]

[REPONSE]
Texte complet de ton intervention.
[/REPONSE]
""",
    input_variables=["summary", "history", "context", "question"]
)

async def generate_speech(text, file_path):
    communicate = edge_tts.Communicate(text, "fr-FR-HenriNeural")
    await communicate.save(file_path)

# ==============================================================================
# BARRE LATÉRALE : TABLEAU DE BORD DE L'INCIDENT
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/ABB_logo.svg/320px-ABB_logo.svg.png", width=140)
    st.title("🎛️ Poste de Commande")
    st.markdown("---")
    
    session_id = st.text_input("🏷️ Machine / Poste :", value="ACS880_ATELIER_1")
    
    if "current_session" not in st.session_state or st.session_state["current_session"] != session_id:
        st.session_state["current_session"] = session_id
        saved_msgs, saved_summary, updated_at = load_session_from_disk(session_id)
        st.session_state.messages = saved_msgs
        st.session_state["incident_summary"] = saved_summary
        st.session_state["last_active"] = updated_at

    if st.session_state.get("last_active"):
        elapsed_min = int((time.time() - st.session_state["last_active"]) / 60)
        remaining_min = max(0, int((SESSION_TTL_SECONDS - (time.time() - st.session_state["last_active"])) / 60))
        st.success(f"🟢 **Session active (Mémoire 4H)**\n- Dernière action : il y a {elapsed_min} min\n- Temps restant : **{remaining_min} min**")
    else:
        st.info("ℹ️ Nouvelle session (durée : 4 heures)")

    st.markdown("---")
    st.markdown("### 📋 Carnet de Bord d'Incident")
    st.markdown(f"""
    <div class="incident-card">
        {st.session_state.get('incident_summary', 'En attente de signalement...')}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Réinitialiser le diagnostic"):
        delete_session_from_disk(session_id)
        st.session_state.messages = []
        st.session_state["incident_summary"] = "En attente de signalement..."
        st.session_state["last_active"] = None
        st.rerun()

# ==============================================================================
# ZONE DE CONVERSATION
# ==============================================================================
st.markdown('<div class="main-header">ABB ACS880 - Expert Diagnostic Haute Fidélité</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diagnostic interactif assisté par l\'IA avec protocole de questions fermées (OUI / NON)</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio_path" in msg and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"])

# ==============================================================================
# BOUTONS D'ACTION RAPIDE (OUI / NON) & ENTRÉES
# ==============================================================================
st.markdown("---")
st.markdown("##### ⚡ Réponse rapide pour le technicien :")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
quick_answer = None

with col_btn1:
    if st.button("✅ OUI à tout / OUI", use_container_width=True):
        quick_answer = "Oui, j'ai vérifié et c'est confirmé."

with col_btn2:
    if st.button("❌ NON / Aucun défaut", use_container_width=True):
        quick_answer = "Non, ce n'est pas le cas."

with col_btn3:
    if st.button("⚠️ Oui au code 3220 (Sous-tension)", use_container_width=True):
        quick_answer = "Oui, le variateur affiche exactement le code défaut 3220."

# Entrée Vocale
st.markdown("##### 🎤 Ou répondre à la voix :")
audio_val = st.audio_input("Enregistrer votre réponse vocale")

# Entrée Texte classique en bas
text_input = st.chat_input("Ou tapez votre réponse détaillée ici...")

user_prompt = None

if quick_answer:
    user_prompt = quick_answer
elif audio_val is not None:
    with st.spinner("🎙️ Transcription de la voix..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_val.read())
            tmp_path = tmp.name
        with open(tmp_path, "rb") as af:
            transcript = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(tmp_path, af.read()),
                language="fr"
            )
        user_prompt = transcript.text
        os.remove(tmp_path)
elif text_input:
    user_prompt = text_input

# ==============================================================================
# EXÉCUTION DU DIAGNOSTIC
# ==============================================================================
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(f"**👨‍🔧 Technicien :** {user_prompt}")

    with st.chat_message("assistant"):
        status = st.status("🧠 Analyse en cours avec le manuel ABB...", expanded=False)
        
        history_str = ""
        for m in st.session_state.messages[-4:-1]:
            history_str += f"{m['role'].upper()}: {m['content']}\n"
            
        current_summary = st.session_state.get("incident_summary", "Début de panne.")
        
        # Reformulation
        search_query = llm.invoke(query_rewrite_prompt.format(
            summary=current_summary,
            question=user_prompt
        )).content.strip()
        status.write(f"🔍 Recherche ciblée : *\"{search_query}\"*")
        
        # Retrieval
        docs = vector_db.similarity_search(search_query, k=4)
        context_str = ""
        for d in docs:
            p = d.metadata.get('page', 0) + 1
            src = d.metadata.get('source_type', 'Manuel')
            context_str += f"\n--- [{src} - Page {p}] ---\n{d.page_content}\n"
            
        # Modèle 120B
        full_res = llm.invoke(diagnostic_prompt.format(
            summary=current_summary,
            history=history_str if history_str else "Nouveau cas.",
            context=context_str,
            question=user_prompt
        )).content
        
        # Extraction propre avec le nouveau parseur
        extracted_summary, vocal_text, agent_reply = parse_llm_output(full_res)

        new_summary = extracted_summary if extracted_summary else current_summary
        st.session_state["incident_summary"] = new_summary
        st.session_state["last_active"] = time.time()
        
        # Audio
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        asyncio.run(generate_speech(vocal_text if vocal_text else agent_reply[:140], audio_file))
        
        status.update(label="Directive prête", state="complete")
        
        st.markdown(agent_reply)
        st.markdown("📢 **Consigne vocale :**")
        st.audio(audio_file)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": agent_reply,
            "audio_path": audio_file
        })
        
        # Sauvegarde disque (4h)
        save_session_to_disk(
            session_id=st.session_state["current_session"],
            messages=st.session_state.messages,
            summary=new_summary
        )
        st.rerun()