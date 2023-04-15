# LaMa_recoloration
 Correction d'images décolorées obtenues par l'algorithme d'inpainting LaMa
 
## Pour la méthode 1 (dossier "LaMa_inpainting_correction_ViT") :

(1) installer les librairies requises (pip install -r requirements.txt)
(2) Dans inpainting_fix.py, préciser "root", le nom de l'image "imgname", le nom du masque "maskname" et les hyperparamètres
(3) lancer inpainting_fix.py

## Pour la methode 2 (dossier "LaMa_inpainting_correction_Gatys") :
(1) Configuration de l'environnement
    ```
    cd LaMa_inpainting_correction_Gatys
    pip install -r requirements.txt 
    ```

(2) Exécuter l'algorithme sur la sortie de LaMa
    ```
    python3 gatys_inpainting.py --image_path=image.png --mask_path=mask.png
    ```

 
