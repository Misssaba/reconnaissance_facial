import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc(y_true, scores):
    # Conversion en tableaux numpy pour assurer la compatibilité des calculs
    y_true = np.array(y_true)
    scores = np.array(scores)
    # Calcul des taux de faux positifs (fpr) et vrais positifs (tpr) pour différents seuils
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Calcul de l'AUC (Area Under the Curve) : plus il est proche de 1.0, plus le modèle est performant
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def best_threshold_eer(y_true, scores):
    # Récupération des taux fpr (FAR) et tpr (vrais positifs)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Calcul du False Rejection Rate (FRR) : c'est l'inverse du taux de vrais positifs
    fnr = 1 - tpr
    # Recherche de l'indice où la différence entre FAR (fpr) et FRR (fnr) est minimale
    # On utilise np.absolute pour trouver le point de croisement le plus proche de zéro
    idx = np.nanargmin(np.absolute(fpr - fnr))
    # Retourne le seuil idéal ainsi que les taux d'erreur à ce point
    return thresholds[idx], fpr[idx], fnr[idx]