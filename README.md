<<<<<<< HEAD
# Système de Reconnaissance Faciale : FaceNet vs ArcFace

Ce projet est une plateforme de biométrie faciale interactive développée avec Streamlit. Il permet de comparer l'efficacité des modèles FaceNet et ArcFace pour la vérification et l'identification d'identités.

## Fonctionnalités principales

Le système est organisé en trois modules métiers :

1. **Évaluation des Performances (Benchmark)** :
   - Test automatisé sur le dataset LFW (Labeled Faces in the Wild).
   - Calcul des métriques de précision : AUC (Area Under Curve), EER (Equal Error Rate), FAR et FRR.
   - Comparaison visuelle via les courbes ROC.

2. **Vérification 1:1** :
   - Comparaison de deux visages via Webcam ou téléchargement de fichiers.
   - Calcul de la similarité cosinus avec seuil de décision ajustable.
   - Confirmation d'identité instantanée.

3. **Identification 1:N** :
   - Enregistrement d'utilisateurs dans une base de données SQLite.
   - Recherche du "plus proche voisin" dans la base de données pour identifier un inconnu.
   - Gestion de la base (affichage et réinitialisation).

##  Stack Technique

- **Langage** : Python 3.11
- **Deep Learning** : PyTorch, Facenet-PyTorch
- **Modèles** : InceptionResnetV1 (FaceNet & ArcFace)
- **Détection & Alignement** : MTCNN (Multi-task Cascaded Convolutional Networks)
- **Interface** : Streamlit
- **Base de données** : SQLite (Stockage des vecteurs d'embeddings 512D)

## Installation et Configuration

1. **Clonage du dépôt** :
   ```bash
   git clone https://github.com/Misssaba/reconnaissance_facial.git
   cd reconnaissance-faciale
 ## Création de l'environnement virtuel 
    python -m venv venv
    # Activation (Windows) : venv\Scripts\activate
    # Activation (Mac/Linux) : source venv/bin/activate

 ## Installation des dépendances
 pip install -r requirements.txt

 ## Lancer l'application

 streamlit run app.py

## Structure du Dossier
```
├── app.py                # Interface principale (Frontend)
├── db.py                 # Gestionnaire de base de données SQLite
├── src/
│   ├── embeddings.py     # Moteur d'extraction (Logic MTCNN + Modèles)
│   ├── metrics.py        # Algorithmes de calcul ROC, AUC, EER
│   └── utils.py          # Utilitaires (Similarité Cosinus, etc.)
├── requirements.txt      # Dépendances Python
└── README.md             # Documentation du projet

