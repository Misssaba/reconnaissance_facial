# src/db.py
import sqlite3
import numpy as np
import json
from pathlib import Path
from src.utils import cosine_similarity

# Définition du chemin de la base de données SQLite
DB_PATH = Path(__file__).parent.parent / "faces.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Création de la table 'users'
    # name : Nom de la personne
    # embedding : Signature faciale stockée sous forme de texte (JSON)
    # enrollment_date : Date d'enregistrement automatique
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_user(name, embedding):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Conversion du vecteur NumPy en liste Python, puis en chaîne JSON pour le stockage SQL
    embedding_list = embedding.tolist() 
    embedding_json = json.dumps(embedding_list)
    
    cursor.execute(
        "INSERT INTO users (name, embedding) VALUES (?, ?)",
        (name, embedding_json)
    )
    conn.commit()
    user_id = cursor.lastrowid # Récupère l'ID généré pour le nouvel utilisateur
    conn.close()
    return user_id

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM users")
    
    users = []
    for row in cursor.fetchall():
        # Désérialisation du JSON pour retrouver le vecteur NumPy original
        emb_list = json.loads(row[2])
        emb_array = np.array(emb_list) 
        users.append({
            "id": row[0], 
            "name": row[1], 
            "embedding": emb_array
        })
    conn.close()
    return users

def find_closest_match(query_embedding, threshold=0.6):
    users = get_all_users()
    if not users:
        return None, 0.0
        
    best_match = None
    best_score = -1.0

    # Boucle de comparaison (Matching)
    for user in users:
        # Calcul de la ressemblance entre le visage actuel et le visage en base
        score = cosine_similarity(query_embedding, user["embedding"])
        
        # On mémorise le score le plus élevé (le visage le plus proche)
        if score > best_score:
            best_score = score
            best_match = user
            
    # Si même le meilleur score est trop bas, on considère la personne comme "Inconnue"
    if best_score < threshold:
        best_match = None

    return best_match, best_score