import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Charger le modèle de segmentation YOLOv8n
model = YOLO("yolov8n-seg.pt")  

# Récupérer les noms des classes du modèle
names = model.model.names

# Ouvrir la vidéo à partir du fichier spécifié
cap = cv2.VideoCapture("chemin/vers/le/fichier/video.mp4")

# Obtenir la largeur (w), la hauteur (h) et le nombre d'images par seconde (fps) de la vidéo
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Créer un objet VideoWriter pour écrire la vidéo de sortie
out = cv2.VideoWriter('segmentation-instance.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    # Lire une image de la vidéo
    ret, im0 = cap.read()
    if not ret:
        print("La trame vidéo est vide ou le traitement de la vidéo est terminé avec succès.")
        break

    # Effectuer la prédiction de segmentation sur l'image
    results = model.predict(im0)
    
    # Créer un annotateur pour ajouter des boîtes de délimitation et des masques sur l'image
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        # Récupérer les masques prédits et les classes associées
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        
        # Ajouter les boîtes de délimitation et les masques sur l'image
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])

    # Écrire l'image annotée dans la vidéo de sortie
    out.write(im0)
    
    # Afficher l'image annotée
    cv2.imshow("segmentation-instance", im0)

    # Attendre la touche 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources utilisées
out.release()
cap.release()
cv2.destroyAllWindows()
