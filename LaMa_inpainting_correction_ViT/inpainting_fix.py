#%% Importations

import cv2
from PIL import Image
import numpy as np
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from Rabin_peyre import transfert_couleurs
from tqdm import tqdm
import os

#%% Dossier "root"

root = '/home/onyxia/lama_inpainting_correction_ViT'

#%% Nom de l'image (contenue dans "root") à corriger

imgname = '/bertrand-gabioud-CpuFzIsHYJ0.png'

#%% Nom du masque (contenu dans "root")

imgmask = 'bertrand-gabioud-CpuFzIsHYJ0_mask.png'

#%% Création des dossiers submasks et best_patches
#%% submasks contient les sous-masques créés
#%% best_patches contient le meilleur patch pour chaque élément de submasks

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"'{folder_name}' créé")
    else:
        print(f"'{folder_name}' existe déjà")

def count_elements_in_folder(folder_name):
    num_elements = 0
    
    if os.path.exists(folder_name):
        folder_contents = os.listdir(folder_name)
        num_elements = len(folder_contents)
    else:
        print(f"'{folder_name}' does not exist.")
        
    return num_elements

create_folder(root + '/submasks')
create_folder(root + '/best_patches')

nbre_submasks = count_elements_in_folder(root + '/submasks')

#%% Chargement du masque binaire
mask = Image.open(f"{root}/{maskname}")

#%% Découpage du masque
patch_size = (150,150)
stride_mask = 150

#%% Remplissage du dossier submasks
count = 0
for y in range(0, mask.height - patch_size[1] + 1, stride_mask):
    for x in range(0, mask.width - patch_size[0] + 1, stride_mask):
        #%% si le patch et le masquent s'intersectent
        if mask.crop((x, y, x+patch_size[0], y+patch_size[1])).getbbox():
            submask = Image.new("1", mask.size, 0)

            submask.paste(Image.new("1", (patch_size[0], patch_size[1]), 1), (x, y))

            #%% Création du sous-masque
            submask.save(f"{root}/submasks/mask{count}.png")

            count += 1

#%% Chargement de l'image
image_path = root + imgname

image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float64)
mask_orig = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

#%% Chargement d'un ViT small pré-entraîné sur ImageNet-1k, utilisé pour la sélection des meilleurs patches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = ViTFeatureExtractor.from_pretrained('WinKawaks/vit-small-patch16-224')
model = ViTForImageClassification.from_pretrained('WinKawaks/vit-small-patch16-224', output_hidden_states=True).to(device)
input_size = feature_extractor.size

# Step 3: Compute the L2 distance between the features of the masked region and the features of each rectangular patch around the mask
patch_stride = (10, 10)
features_size = (197, 384)

#%% Précalcul des features des patches hors masque (P_ext)

precomputed_features = np.zeros((image.shape[0], image.shape[1], features_size[0], features_size[1]))

for i in range(patch_size[0], image.shape[0] - patch_size[0], patch_stride[0]):
        for j in range(patch_size[1], image.shape[1] - patch_size[1], patch_stride[1]):
            if mask_orig[i:i+patch_size[0], j:j+patch_size[1]].sum() == 0:
                patch = image[i:i+patch_size[0], j:j+patch_size[1], :]
                resized_patch = cv2.resize(patch, (input_size['height'], input_size['width']))
                inputs = feature_extractor(images=resized_patch.astype(np.uint8), return_tensors="pt").to(device)
                outputs = model(**inputs)
                patch_features = outputs.hidden_states[-1].squeeze().detach().to('cpu').numpy()
                precomputed_features[i][j] = patch_features

#%% Boucle principale : 

for u in tqdm(range(nbre_submasks)):
    mask_path = root + "/submasks/mask" + str(u) + ".png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    #%% Extraction des coordonnées du sous-masque u
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    #%% Récupération des features de la zone à reconstruire, afin de comparer aux patches précalculés
    zone = image[y:y+h, x:x+w, :]
    resized_zone = cv2.resize(zone, (input_size['height'], input_size['width']))
    inputs_zone = feature_extractor(images=resized_zone.astype(np.uint8), return_tensors="pt").to(device)
    outputs_zone = model(**inputs_zone)
    zone_features = outputs_zone.hidden_states[-1].squeeze().detach().to('cpu').numpy()

    #%% Calcul de la matrice des cosine similarities entre les patches de P_ext et les features des zones à reconstruire

    distances = []
    for i in range(patch_size[0], image.shape[0] - patch_size[0], patch_stride[0]):
        for j in range(patch_size[1], image.shape[1] - patch_size[1], patch_stride[1]):
            if mask[i:i+patch_size[0], j:j+patch_size[1]].sum() == 0 and mask_orig[i:i+patch_size[0], j:j+patch_size[1]].sum() == 0:
                distance = cosine_similarity(zone_features.reshape(-1, zone_features.shape[-1]), precomputed_features[i][j].reshape(-1, precomputed_features[i][j].shape[-1])).mean()
                distances.append((i, j, distance))

    #%% Récupération du meilleur patch en termes de cosine similarity
    best_patch = None
    best_distance = -1
    for i, j, distance in distances:
        if distance > best_distance:
            best_patch = image[i:i+patch_size[0], j:j+patch_size[1], :]
            best_distance = distance

    #%% Sauvegarde du meilleur patch
    cv2.imwrite(f'{root}/best_patches/best_patch' + str(u) + '.png', best_patch)
    img_mask_copy = image[y:y+h,x:x+w,:]

    #%% Transfert des couleurs du patch sélectionné à la partie décolorée
    transfert_image = transfert_couleurs(img_mask_copy.astype(np.float32),best_patch.astype(np.float32),nbetapes=100,lmbS=1)

    #%% seamlessClone afin d'incruster l'image recolorée
    src_mask = np.ones(transfert_image.shape, dtype=transfert_image.dtype)*255
    center = (x + w//2, y + h//2)
    result = cv2.seamlessClone(transfert_image.astype('uint8'), image.astype('uint8'), src_mask.astype('uint8'), center, cv2.NORMAL_CLONE)

    image = result.astype(np.uint8)

#%% Sauvegarde de l'image finale
cv2.imwrite(f'{root}/img_corrected.png', image)