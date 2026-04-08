import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from src.embeddings import EmbeddingExtractor
from src.metrics import compute_roc, best_threshold_eer
from src.utils import cosine_similarity
from db import init_db, add_user, get_all_users, find_closest_match
from sklearn.datasets import fetch_lfw_pairs

# Configuration de la page Streamlit (Titre et mise en page large)
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("Système Reconnaissance Faciale")

# Initialisation de la base de données SQLite au démarrage
init_db()

extractor = EmbeddingExtractor()
# Organisation de l'interface en 3 onglets (Tabs)
tab1, tab2,tab3 = st.tabs([ "Evaluation","VERIFICATION (deux images)", "IDENTIFICATION(Base de données)"])

@st.cache_resource
def get_extractor():
    """Charge l'extracteur de caractéristiques."""
    return EmbeddingExtractor()

@st.cache_data(persist=False)
def load_data():
    """Charge le dataset LFW (Labeled Faces in the Wild) pour l'évaluation."""
    # resize=1.0 pour garder la résolution originale avant le traitement IA
    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    return data.pairs, data.target

# Instanciation de l'extracteur et chargement des données
extractor = get_extractor()
pairs, y_true = load_data()

def clear_database():
    """Supprime physiquement la base de données et la réinitialise."""
    from db import DB_PATH
    import os
    
    if DB_PATH.exists():
        os.remove(DB_PATH)
        st.success(f"Base de données supprimée")
    init_db()
 # --- ONGLET 1 : ÉVALUATION DES PERFORMANCES ---   
with tab1: 
    st.subheader("Évaluation LFW")
    if st.button("Lancer l’évaluation"):
        with st.spinner('Évaluation en cours... '):
            scores_facenet = []
            scores_arcface = []
            y = []
            # Sélection de 1000 paires équilibrées (1000 positives, 1000 négatives)
            pos_idx = np.where(y_true == 1)[0][:1000]
            neg_idx = np.where(y_true == 0)[0][:1000]
            balanced_idx = np.concatenate([pos_idx, neg_idx])
            for i in balanced_idx:
                pair = pairs[i]
                label = y_true[i]
                # Conversion des données Sklearn en format Image PIL
                img_a = Image.fromarray((pair[0] * 255).astype(np.uint8))
                img_b = Image.fromarray((pair[1] * 255).astype(np.uint8))
                # Extraction et calcul de similarité pour FaceNet et ArcFace
                e1 = extractor.extract(img_a, "facenet")
                e2 = extractor.extract(img_b, "facenet")
                scores_facenet.append(cosine_similarity(e1, e2))
                e3 = extractor.extract(img_a, "arcface")
                e4 = extractor.extract(img_b, "arcface")
                scores_arcface.append(cosine_similarity(e3, e4))
                y.append(int(label))
            # Calcul des métriques statistiques
            y = np.array(y)
            scores_facenet = np.array(scores_facenet)
            scores_arcface = np.array(scores_arcface)
            thr_f, far_f, frr_f = best_threshold_eer(y, scores_facenet)
            thr_a, far_a, frr_a = best_threshold_eer(y, scores_arcface)
            fpr_f, tpr_f, _, auc_f = compute_roc(y, scores_facenet)
            fpr_a, tpr_a, _, auc_a = compute_roc(y, scores_arcface)

	    # Affichage de la courbe ROC
            fig, ax = plt.subplots()
            ax.plot(fpr_f, tpr_f, label=f"FaceNet AUC={auc_f:.3f}")
            ax.plot(fpr_a, tpr_a, label=f"ArcFace AUC={auc_a:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
	    # Affichage du tableau récapitulatif
            df = pd.DataFrame([
                {"Model": "FaceNet", "Threshold": thr_f, "FAR": far_f, "FRR": frr_f, "AUC": auc_f},
                {"Model": "ArcFace", "Threshold": thr_a, "FAR": far_a, "FRR": frr_a, "AUC": auc_a},
            ])
            st.dataframe(df)

# --- ONGLET 2 : VÉRIFICATION 1:1 (Comparaison de deux visages) ---

with tab2:
    st.header("VÉRIFICATION : Comparer 2 visages")
    st.sidebar.header("Paramètres")
    model_choice = st.sidebar.selectbox("Modèle", ["facenet", "arcface"])
    threshold = st.sidebar.slider("Seuil", 0.0, 1.0, 0.6, 0.01)
    mode_acq = st.radio("Acquisition", ["Upload", "Webcam"],key="radio_1:1")

    # Gestion de l'acquisition (soit fichiers, soit caméra directe)
    if mode_acq == "Upload":
        col1, col2 = st.columns(2)
        with col1: img1_file = st.file_uploader("Image 1", type=["jpg","png","jpeg"])
        with col2: img2_file = st.file_uploader("Image 2", type=["jpg","png","jpeg"])
        
        if img1_file and img2_file:
            img1 = Image.open(img1_file).convert("RGB")
            img2 = Image.open(img2_file).convert("RGB")
            st.image([img1, img2], width=220)
            e1 = extractor.extract(img1, model_choice)
            e2 = extractor.extract(img2, model_choice)
            score = cosine_similarity(e1, e2)
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{score:.4f}")
            col2.metric("Seuil", f"{threshold:.4f}")
            if score >= threshold:
                st.success("MÊME IDENTITÉ")
            else:
                st.error("Différentes")
    elif mode_acq == "Webcam":
        col1, col2 = st.columns(2)
        with col1: cam1 = st.camera_input("Photo 1")
        with col2: cam2 = st.camera_input("Photo 2")
        
        if cam1 and cam2:
            img1 = Image.open(cam1).convert("RGB")
            img2 = Image.open(cam2).convert("RGB")
            st.image([img1, img2], width=220)
            
            e1 = extractor.extract(img1, model_choice)
            e2 = extractor.extract(img2, model_choice)
            score = cosine_similarity(e1, e2)
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{score:.4f}")
            col2.metric("Seuil", f"{threshold:.4f}")
              # Affichage des métriques de résultat
            if score >= threshold:
                st.success("MÊME IDENTITE")
            else:
                st.error("Différentes")

# --- ONGLET 3 : IDENTIFICATION 1:N (Recherche en base de données) ---
with tab3:
    st.header("IDENTIFICATION : Recherche dans la base de données")
    with st.sidebar.header("Paramètres 1:N"):
        model_1n = st.sidebar.selectbox("Modèle 1:N", ["facenet", "arcface"])
        threshold_1n = st.sidebar.slider("Seuil 1:N", 0.0, 1.0, 0.6, 0.01)
    tab4, tab5 = st.tabs(["Enregistrement", "Identification"])
    
    with tab4:
        st.subheader("Ajouter dans la base de données")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nom:")
            enroll_file = st.file_uploader("Photo enrôlement", key="enroll")
        with col2:
            if enroll_file and name and st.button("Enregistrer"):
                img = Image.open(enroll_file).convert("RGB")
                embedding = extractor.extract(img, model_1n)
                user_id = add_user(name, embedding)
                st.success(f"{name} enregistré (ID: {user_id})")
        st.subheader("Base utilisateurs")
        if st.button("Afficher la liste des utilisateurs"):
            users = get_all_users()
            if users:
                for user in users:
                    st.write(f"• **{user['name']}** ")
            else:
                st.info("Aucune personne enregistrée")
        if st.button("EFFACER TOUTE LA BASE DE DONNÉES"):
            clear_database()
            st.success("Base de données complètement effacée !")
            st.rerun()
        
    with tab5:             
        st.subheader("Identification")
        query_mode = st.radio("Acquisition", ["Upload", "Webcam"],key="radio_1:N")
        
        if query_mode == "Upload":
            query_file = st.file_uploader("Image à identifier")
            if query_file:
                img = Image.open(query_file).convert("RGB")
                st.image(img, width=200)
                
                emb = extractor.extract(img, model_1n)

                match, score = find_closest_match(emb, threshold_1n)
                if match:
                    st.success(f" **{match['name']}** (score: {score:.3f})")
                else:
                    st.warning(f"Inconnu (score: {score:.3f})")
                    st.write("Embedding de la requête (premiers 5 valeurs):", emb[:5])
        elif query_mode == "Webcam":
            cam = st.camera_input("Photo webcam")
            if cam:
                img = Image.open(cam).convert("RGB")
                st.image(img, width=200)
                
                emb = extractor.extract(img, model_1n)
            # Recherche du match le plus proche dans la base SQLite
                match, score = find_closest_match(emb, threshold_1n)
                
                if match:
                    st.success(f" **{match['name']}** (score: {score:.3f})")
                else:
                    st.warning(f"Inconnu (score: {score:.3f})")
 
