import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Vérification du fichier
pdf_path = "data/manual.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ Erreur : Le fichier '{pdf_path}' est introuvable !")
    print("Vérifiez que votre PDF est bien placé dans le dossier 'data' et nommé 'manual.pdf'.")
    exit()

print("--- Étape 1 : Lecture du manuel ABB ACS880 ---")
loader = PyPDFLoader(pdf_path)

# Pour ce premier test rapide, on charge les 40 premières pages (sommaire, consignes de sécurité, aperçu matériel)
all_pages = loader.load()
pages_to_test = all_pages[:40] 
print(f"✅ {len(pages_to_test)} pages chargées avec succès.")

# 2. Découpage en blocs (Chunking)
print("\n--- Étape 2 : Découpage du texte en segments exploitables ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # blocs de 600 caractères
    chunk_overlap=100     # chevauchement pour ne pas couper le sens
)
chunks = text_splitter.split_documents(pages_to_test)
print(f"✅ {len(chunks)} blocs de texte créés.")

# 3. Création des vecteurs d'embeddings (Modèle compact et très rapide)
print("\n--- Étape 3 : Conversion mathématique en vecteurs (Embeddings) ---")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Stockage dans ChromaDB
print("--- Étape 4 : Indexation dans la base vectorielle ---")
vector_db = Chroma.from_documents(chunks, embeddings_model)
print("✅ Base vectorielle prête !")

# 5. Question type génie électrique / technicien de maintenance
question = "What are the safety instructions regarding capacitor discharge and electrical shock?"
print(f"\n=======================================================")
print(f"❓ QUESTION TECHNICIEN : {question}")
print(f"=======================================================")

# Recherche des 2 blocs les plus pertinents
resultats = vector_db.similarity_search(question, k=2)

for i, doc in enumerate(resultats):
    page_num = doc.metadata.get('page', 0) + 1  # +1 car l'index commence à 0
    print(f"\n🔎 [RÉSULTAT {i+1}] - Trouvé à la PAGE {page_num} du manuel :")
    print("-" * 50)
    print(doc.page_content.strip())
    print("-" * 50)