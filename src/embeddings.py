import torch # Framework deep learning pour la gestion des tenseurs et modèles
import torch.nn.functional as F  # Fonctions utilitaires (normalisation, interpolation)
from PIL import Image # Bibliothèque standard pour le traitement d'images
from torchvision import transforms  # Transformations pour convertir les images en tenseurs
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization 

class EmbeddingExtractor:
    def __init__(self, device=None):
        # Sélection du processeur 
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # MTCNN : Détecteur de visages. Il localise, aligne et recadre le visage.
        self.detector = MTCNN(
            image_size=160, # Taille de sortie standard pour FaceNet
            margin=20,      # Ajoute une marge autour du visage détecté
            device=self.device,
            post_process=False # Garde les pixels bruts avant la standardisation manuelle
        )
        
        # Chargement de FaceNet (pré-entraîné sur VGGFace2) 
        self.model_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Chargement d'ArcFace (architecture Inception pré-entraînée sur CASIA-WebFace)
        self.model_arcface = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)

    @torch.no_grad() # Désactive le calcul des gradients pour gain de mémoire et de vitesse en inférence
    def extract(self, img, model_name="facenet", use_detector=True):
        # Conversion de l'image en format PIL (format attendu par MTCNN et transforms)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        face = None
        # Utilisation du détecteur MTCNN pour isoler le visage
        if use_detector:
            face = self.detector(img)
        
        # Gestion des cas sans détection 
        if face is None:
            # On redimensionne manuellement selon le modèle choisi
            size = 160 if model_name == "facenet" else 112
            face = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor() # Conversion en tenseur PyTorch [0, 1]
            ])(img)
        elif model_name == "arcface":
            # ArcFace attend du 112x112, on redimensionne si MTCNN a sorti du 160x160
            face = F.interpolate(face.unsqueeze(0), size=(112, 112), mode='bilinear').squeeze(0)

        # Standardisation des pixels pour correspondre à l'entraînement
        x = fixed_image_standardization(face).unsqueeze(0).to(self.device)
        
        # Choix du modèle et passage de l'image 
        model = self.model_facenet if model_name == "facenet" else self.model_arcface
        emb = model(x) 
        # Normalisation pour que le produit scalaire soit égal à la similarité cosinus.
        emb = F.normalize(emb, p=2, dim=1)
        # Retourne un vecteur Numpy 1D 
        return emb.squeeze(0).cpu().numpy()