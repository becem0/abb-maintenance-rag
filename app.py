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
import base64
import sqlite3
import asyncio
import tempfile
import threading
import pygame
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from rank_bm25 import BM25Okapi

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

print("=" * 70)
print("⚡ ABB ACS880 COGNITIVE MAINTENANCE AGENT // PROMOTION ENIM 2026")
print("👨‍💻 CONCEPTION & DÉVELOPPEMENT : Ameur Bacem (Génie Électrique)")
print("🔒 PROPRIÉTÉ INTELLECTUELLE PROTÉGÉE - ENIM")
print("=" * 70)

st.set_page_config(
    page_title="ABB ACS880 - Cockpit Diagnostic RAG Hybride",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FF000F !important;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569 !important;
        margin-bottom: 16px;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        margin-bottom: 14px !important;
        padding: 16px !important;
    }
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 15px 0 !important;
    }
    th {
        background-color: #f1f5f9 !important;
        color: #FF000F !important;
        font-weight: 700 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 10px !important;
    }
    td {
        border: 1px solid #e2e8f0 !important;
        padding: 9px !important;
        font-size: 0.95rem !important;
    }
    .white-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #FF000F;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .stButton>button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        border-color: #FF000F !important;
        color: #FF000F !important;
        box-shadow: 0 2px 8px rgba(255, 0, 15, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BASE SQLITE (MÉMOIRE 4H AVEC AUTO-MIGRATION)
# ==============================================================================
DB_FILE = "maintenance_memory.db"
SESSION_TTL_SECONDS = 4 * 3600

def init_memory_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS project_author (
            author_name TEXT,
            school TEXT,
            academic_year TEXT,
            signature_hash TEXT
        )
    """)
    c.execute("INSERT OR IGNORE INTO project_author VALUES ('Ameur Bacem', 'ENIM Monastir', '2025-2026', 'ENIM-GE-2026-AUTOR-01')")
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

def save_session(session_id, messages, summary):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO sessions (session_id, updated_at, messages_json, incident_summary)
        VALUES (?, ?, ?, ?)
    """, (session_id, time.time(), json.dumps(messages), summary))
    conn.commit()
    conn.close()

def load_session(session_id):
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
            return [], "Session expirée après 4h d'inactivité.", None
    return [], "En attente de signalement de panne...", None

init_memory_db()

# ==============================================================================
# MOTEUR RAG HYBRIDE EN CACHE (CHROMA VECTEURS + BM25 MOTS-CLÉS)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_hybrid_rag_engine():
    DB_DIR = "chroma_db_abb_multi"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Construction de l'index BM25 sur tous les documents pour la recherche exacte
    all_data = vector_db.get()
    corpus_texts = all_data['documents']
    corpus_metas = all_data['metadatas']
    
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    groq_client = Groq()


    return vector_db, bm25, corpus_texts, corpus_metas, llm, groq_client

with st.spinner("⚡ Initialisation du Réacteur RAG Hybride (Vecteurs + BM25 + Modèle 120B)..."):
    vector_db, bm25, corpus_texts, corpus_metas, llm, groq_client = load_hybrid_rag_engine()

# ==============================================================================
# ALGORITHME DE RECHERCHE HYBRIDE RRF (RECIPROCAL RANK FUSION)
# ==============================================================================
def hybrid_retrieval_rrf(query_en, top_k=4):
    """
    Fusionne mathématiquement la recherche vectorielle sémantique et la recherche
    par mot-clé exact BM25 via l'algorithme RRF officiel.
    """
    # 1. Recherche vectorielle sémantique (Top 6)
    vector_results = vector_db.similarity_search(query_en, k=6)
    
    # 2. Recherche par mots-clés exacts BM25 (Top 6)
    tokenized_q = query_en.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:6]
    
    # 3. Calcul des scores RRF
    rrf_scores = {}
    docs_map = {}
    
    # RRF sur les vecteurs
    for rank, doc in enumerate(vector_results):
        txt = doc.page_content
        rrf_scores[txt] = rrf_scores.get(txt, 0.0) + (1.0 / (60.0 + rank))
        docs_map[txt] = doc
        
    # RRF sur BM25
    for rank, idx in enumerate(top_bm25_idx):
        if bm25_scores[idx] > 0:
            txt = corpus_texts[idx]
            meta = corpus_metas[idx] if corpus_metas else {}
            rrf_scores[txt] = rrf_scores.get(txt, 0.0) + (1.0 / (60.0 + rank))
            if txt not in docs_map:
                docs_map[txt] = Document(page_content=txt, metadata=meta)
                
    # Tri décroissant selon le score RRF combiné
    sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)
    return [docs_map[t] for t in sorted_texts[:top_k]]

# ==============================================================================
# ENCODAGE DU JUMEAU 3D
# ==============================================================================
def get_base64_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

b64_glb = get_base64_file("data/acs880.glb")
b64_closed = get_base64_file("data/acs880_closed.png")

def render_compact_3d_twin():
    """Affiche le variateur ABB sans espace blanc inutile grâce à bounds='tight'."""
    if b64_glb:
        model_src = f"data:model/gltf-binary;base64,{b64_glb}"
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
            <style>
                body {{ margin: 0; background: #ffffff; overflow: hidden; }}
                model-viewer {{
                    width: 100%; height: 320px;
                    background: radial-gradient(circle at center, #ffffff 0%, #f1f5f9 100%);
                    border: 1px solid #e2e8f0; border-radius: 10px;
                }}
                .hud {{
                    position: absolute; top: 8px; left: 10px; z-index: 10;
                    font-size: 10px; color: #FF000F; font-weight: bold; font-family: sans-serif;
                    background: rgba(255,255,255,0.92); padding: 4px 8px;
                    border-left: 3px solid #FF000F; border-radius: 4px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                }}
            </style>
        </head>
        <body>
            <div class="hud">ABB ACS880 // 3D DIGITAL TWIN [ROTATION CONTINUE]</div>
            <model-viewer src="{model_src}" 
                          camera-controls 
                          auto-rotate 
                          rotation-per-second="20deg"
                          bounds="tight"
                          camera-target="auto auto auto"
                          shadow-intensity="1.0" 
                          exposure="1.0" 
                          environment-image="neutral">
            </model-viewer>
        </body>
        </html>
        """
    else:
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ margin: 0; background: #ffffff; overflow: hidden; font-family: sans-serif; }}
                .stage {{
                    width: 100%; height: 320px; display: flex; align-items: center; justify-content: center;
                    position: relative; background: radial-gradient(circle at center, #ffffff 0%, #f1f5f9 100%);
                    border: 1px solid #e2e8f0; border-radius: 10px;
                }}
                .drive-turntable {{
                    max-height: 90%; max-width: 90%; object-fit: contain;
                    animation: turntable 14s infinite linear;
                }}
                @keyframes turntable {{
                    0% {{ transform: rotateY(0deg); }}
                    50% {{ transform: rotateY(18deg) scale(1.02); }}
                    100% {{ transform: rotateY(0deg); }}
                }}
                .hud {{
                    position: absolute; top: 8px; left: 10px;
                    font-size: 10px; color: #FF000F; font-weight: bold;
                    background: rgba(255,255,255,0.92); padding: 4px 8px;
                    border-left: 3px solid #FF000F; border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="stage">
                <div class="hud">ABB ACS880 // DIGITAL TWIN HD [ROTATION ACTIVE]</div>
                <img class="drive-turntable" src="data:image/png;base64,{b64_closed}" alt="ABB ACS880">
            </div>
        </body>
        </html>
        """
    # Hauteur réduite de 395px à 330px pour supprimer l'espace sous le cadre
    components.html(html_code, height=330, scrolling=False)

# ==============================================================================
# SYSTÈME VOCAL DOUBLE ACTION
# ==============================================================================
async def generate_speech(text, file_path):
    communicate = edge_tts.Communicate(text, "fr-FR-HenriNeural")
    await communicate.save(file_path)

# ==============================================================================
# PARSEUR ROBUSTE
# ==============================================================================
def parse_clean_output(raw_text):
    s_match = re.search(r'\[SYNTHESE_INCIDENT\](.*?)(\[/SYNTHESE_INCIDENT\]|$)', raw_text, re.DOTALL)
    summary = s_match.group(1).strip() if s_match else "Incident en cours d'analyse."

    v_match = re.search(r'\[VOCAL\](.*?)(\[/VOCAL\]|$)', raw_text, re.DOTALL)
    vocal = v_match.group(1).strip() if v_match else ""

    clean_text = raw_text
    clean_text = re.sub(r'\[SYNTHESE_INCIDENT\].*?(\[/SYNTHESE_INCIDENT\]|\n)', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\[VOCAL\].*?(\[/VOCAL\]|\n)', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'\[/?(SYNTHESE_INCIDENT|VOCAL|REPONSE)\]', '', clean_text, flags=re.DOTALL).strip()

    if not vocal:
        sentences = re.split(r'[.!?]', clean_text)
        vocal = ". ".join([s.strip() for s in sentences[:2] if len(s.strip()) > 10]) + "."

    return summary, vocal, clean_text

query_rewrite_prompt = PromptTemplate(
    template="""You are an expert ABB ACS880 drives technical search engine.
Analyze the incident context and technician's message.
Output ONLY an English technical search query targeting the ABB Hardware and Firmware manuals.
Include exact fault codes, terminal designations (e.g. 3210, 5080, IN1, UDC+, U1 V1 W1) if relevant.
INCIDENT: {summary}
TECHNICIAN: {question}
ENGLISH QUERY:""",
    input_variables=["summary", "question"]
)

diagnostic_prompt = PromptTemplate(
    template="""Tu es un ingénieur expert senior en maintenance des variateurs industriels ABB ACS880.
CARNET D'INCIDENT : {summary}
HISTORIQUE : {history}
EXTRAITS MANUELS (OBTENUS PAR RECHERCHE HYBRIDE BM25 + VECTEURS) :
{context}

TECHNICIEN : {question}

CONSIGNES :
1. Si le problème est FLOU : Pose 2 ou 3 questions directes OUI / NON pour éliminer les causes.
2. Si la situation est CIBLÉE : Rédige le rapport complet (Sécurité/Consignation, Cause racine, Tableau étape par étape avec bornes, couples en N·m, et les pages citées).
3. Si la donnée est hors périmètre, applique l'abstention stricte sans inventer.

FORMAT OBLIGATOIRE :
[SYNTHESE_INCIDENT]
Synthèse courte en 2 phrases.
[/SYNTHESE_INCIDENT]

[VOCAL]
2 phrases directes et percutantes que tu dis à voix haute au technicien.
[/VOCAL]

[REPONSE]
Texte complet de ta directive affiché à l'écran.
[/REPONSE]
""",
    input_variables=["summary", "history", "context", "question"]
)

# ==============================================================================
# BARRE LATÉRALE AVEC LOGO VECTORIEL ABB INDESTRUCTIBLE
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
        <div style="background:#FF000F; color:#ffffff; font-weight:900; font-size:26px; padding:3px 14px; border-radius:4px; letter-spacing:1px; font-family:sans-serif;">
            ABB
        </div>
        <div style="font-weight:700; font-size:14px; color:#0f172a; line-height:1.2; font-family:sans-serif;">
            MOTION<br><span style="color:#64748b; font-size:11px;">DRIVES SERVICE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Diagnostic Studio")
    st.markdown("---")
    st.info("⚡ **Moteur RAG Hybride Actif**\n- Dense : Plongements vectoriels\n- Sparse : BM25 (Mots-clés exacts)\n- Fusion : Algorithme RRF")
    
    session_id = st.text_input("🏷️ Machine / Poste :", value="ACS880_LIGNE_1")
    
    if "current_session" not in st.session_state or st.session_state["current_session"] != session_id:
        st.session_state["current_session"] = session_id
        saved_msgs, saved_summary, updated_at = load_session(session_id)
        st.session_state.messages = saved_msgs
        st.session_state["incident_summary"] = saved_summary
        st.session_state["last_active"] = updated_at

    if st.session_state.get("last_active"):
        rem_min = max(0, int((SESSION_TTL_SECONDS - (time.time() - st.session_state["last_active"])) / 60))
        st.success(f"🟢 **Mémoire Active (4H)**\n- Temps restant : **{rem_min} min**")

    st.markdown("### 📋 Carnet de Bord d'Incident")
    st.markdown(f"""
    <div class="white-card">
        {st.session_state.get('incident_summary', 'En attente de signalement de panne...')}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Réinitialiser Incident", use_container_width=True):
        conn = sqlite3.connect(DB_FILE)
        conn.cursor().execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        st.session_state.messages = []
        st.session_state["incident_summary"] = "En attente de signalement..."
        st.session_state["mic_key"] = st.session_state.get("mic_key", 0) + 1
        st.rerun()
    # BADGE OFFICIEL D'AUTEUR & PROPRIÉTÉ INTELLECTUELLE
    st.markdown("---")
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #FF000F; padding:10px; border-radius:6px; font-size:11px;">
        <span style="color:#64748b; font-weight:600;">PROJET D'INGÉNIERIE // R&D</span><br>
        <span style="font-size:13px; font-weight:800; color:#0f172a;">Conçu & Développé par :</span><br>
        <span style="font-size:14px; font-weight:800; color:#FF000F;">Ameur Bacem</span><br>
        <span style="color:#475569; font-weight:500;">Génie Électrique — ENIM (2025-2026)</span><br>
        <span style="color:#94a3b8; font-size:9px;">Tous droits réservés © 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MISE EN PAGE : COCKPIT LARGE (72%) & JUMEAU 3D COMPACT (28%)
# ==============================================================================
st.markdown('<div class="main-title">ABB ACS880 // Digital Twin & AI Diagnostic Cockpit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Assistance de maintenance augmentée par RAG Hybride (BM25 + Dense) et IA 120B</div>', unsafe_allow_html=True)

col_twin, col_chat = st.columns([0.8, 2.2], gap="large")

with col_twin:
    st.markdown("##### Jumeau Numérique 3D")
    render_compact_3d_twin()
    st.caption("🖱️ *Rotation continue 360° | Contrôle souris & molette.*")

# ==============================================================================
# COLONNE DE DROITE : GRAND COCKPIT DIAGNOSTIC & ENTRÉES
# ==============================================================================
with col_chat:
    st.markdown("##### 💬 Cockpit Diagnostic & Directives d'Intervention")
    
    total_messages = len(st.session_state.get("messages", []))
    chat_container = st.container(height=440)
    with chat_container:
        for idx, msg in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "audio_path" in msg and os.path.exists(msg["audio_path"]):
                    is_latest = (idx == total_messages - 1) and (msg["role"] == "assistant")
                    st.audio(msg["audio_path"], autoplay=is_latest)

    # 1. DÉTECTION DU NOMBRE DE QUESTIONS POSÉES PAR L'AGENT
    last_bot_msg = ""
    for m in reversed(st.session_state.get("messages", [])):
        if m["role"] == "assistant":
            last_bot_msg = m["content"]
            break
            
    has_questions = ("?" in last_bot_msg) and any(k in last_bot_msg.upper() for k in ["OUI/NON", "OUI / NON", "1.", "2."])
    user_prompt = None

    # 2. BARRE HORIZONTALE UNIQUE, LARGE ET AÉRÉE (DIRECTEMENT SOUS LA DISCUSSION)
    if has_questions:
        num_q = 3 if ("3." in last_bot_msg or "trois" in last_bot_msg.lower()) else 2
        
        with st.container(border=True):
            st.markdown("<div style='font-size:12px; font-weight:700; color:#FF000F; margin-bottom:4px;'>RÉPONSE RAPIDE AUX QUESTIONS DU DIAGNOSTIC :</div>", unsafe_allow_html=True)
            
            if num_q == 2:
                c1, c2, c3 = st.columns([1.5, 1.5, 1.2])
                with c1:
                    ans_q1 = st.radio("**Question 1 :**", ["OUI", "NON"], horizontal=True, key="diag_q1_m2")
                with c2:
                    ans_q2 = st.radio("**Question 2 :**", ["OUI", "NON"], horizontal=True, key="diag_q2_m2")
                with c3:
                    st.write("")
                    if st.button("Valider", use_container_width=True, key="btn_val_2q"):
                        user_prompt = f"Réponses : Question 1 = {ans_q1}, Question 2 = {ans_q2}."
            else:
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
                with c1:
                    ans_q1 = st.radio("**Question 1 :**", ["OUI", "NON"], horizontal=True, key="diag_q1_m3")
                with c2:
                    ans_q2 = st.radio("**Question 2 :**", ["OUI", "NON"], horizontal=True, key="diag_q2_m3")
                with c3:
                    ans_q3 = st.radio("**Question 3 :**", ["OUI", "NON"], horizontal=True, key="diag_q3_m3")
                with c4:
                    st.write("")
                    if st.button("Valider", use_container_width=True, key="btn_val_3q"):
                        user_prompt = f"Réponses : Question 1 = {ans_q1}, Question 2 = {ans_q2}, Question 3 = {ans_q3}."

    # 3. ZONE DE SAISIE VOCALE & TEXTE
    with st.container(border=True):
        col_m, col_t = st.columns([1, 3])
        
        if "mic_key" not in st.session_state:
            st.session_state.mic_key = 0

        with col_m:
            st.markdown("**🎤 Parler à la voix :**")
            audio_val = st.audio_input("Parler au micro", key=f"mic_hybrid_{st.session_state.mic_key}")
            
        with col_t:
            st.markdown("**⌨️ Écrire votre message :**")
            with st.form("input_form", clear_on_submit=True):
                txt = st.text_input("Message écrit :", placeholder="Décrivez la panne ou précisez...")
                if st.form_submit_button("Envoyer la réponse 🚀", use_container_width=True) and txt.strip():
                    user_prompt = txt.strip()
# ==============================================================================
# TRAITEMENT VOCAL ET RESET DU MICRO
# ==============================================================================
if audio_val is not None and not user_prompt:
    audio_bytes = audio_val.read()
    if len(audio_bytes) > 0:
        with st.spinner("🎙️ Transcription de votre voix par Whisper Turbo..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as af:
                transcript = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=(tmp_path, af.read()),
                    language="fr"
                )
            user_prompt = transcript.text
            os.remove(tmp_path)
            st.session_state.mic_key += 1

# ==============================================================================
# PIPELINE RAG HYBRIDE EN ACTION
# ==============================================================================
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.spinner("🧠 Diagnostic RAG Hybride (BM25 + Dense) en cours..."):
        history_str = ""
        for m in st.session_state.messages[-4:-1]:
            history_str += f"{m['role'].upper()}: {m['content']}\n"
            
        current_summary = st.session_state.get("incident_summary", "Début de panne.")
        
        # 1. Reformulation experte
        search_query = llm.invoke(query_rewrite_prompt.format(
            summary=current_summary,
            question=user_prompt
        )).content.strip()
        
        # 2. RETRIEVAL HYBRIDE RRF DANS LES MANUELS
        docs = hybrid_retrieval_rrf(search_query, top_k=4)
        context_str = ""
        for d in docs:
            p = d.metadata.get('page', 0) + 1
            src = d.metadata.get('source_type', 'Manuel')
            context_str += f"\n--- [{src} - Page {p}] ---\n{d.page_content}\n"
            
        # 3. Raisonnement 120B
        full_res = llm.invoke(diagnostic_prompt.format(
            summary=current_summary,
            history=history_str if history_str else "Nouveau cas.",
            context=context_str,
            question=user_prompt
        )).content
        
        extracted_summary, vocal_text, agent_reply = parse_clean_output(full_res)
        new_summary = extracted_summary if extracted_summary else current_summary
        st.session_state["incident_summary"] = new_summary
        st.session_state["last_active"] = time.time()
        
        # Synthèse vocale
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        speech_text = vocal_text if vocal_text else agent_reply[:180]
        asyncio.run(generate_speech(speech_text, audio_file))
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": agent_reply,
            "audio_path": audio_file
        })
        
        save_session(
            session_id=st.session_state["current_session"],
            messages=st.session_state.messages,
            summary=new_summary
        )
        st.rerun()
        # BADGE OFFICIEL D'AUTEUR & PROPRIÉTÉ INTELLECTUELLE
    st.markdown("---")
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid #FF000F; padding:10px; border-radius:6px; font-size:11px;">
        <span style="color:#64748b; font-weight:600;">PROJET D'INGÉNIERIE // R&D</span><br>
        <span style="font-size:13px; font-weight:800; color:#0f172a;">Conçu & Développé par :</span><br>
        <span style="font-size:14px; font-weight:800; color:#FF000F;">Ameur Bacem</span><br>
        <span style="color:#475569; font-weight:500;">Génie Électrique — ENIM (2025-2026)</span><br>
        <span style="color:#94a3b8; font-size:9px;">Tous droits réservés © 2026</span>
    </div>
    """, unsafe_allow_html=True)