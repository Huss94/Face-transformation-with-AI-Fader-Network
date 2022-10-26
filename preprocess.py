import os 
import cv2 as cv
import numpy as np


# Il faut que l'utilisateur ait le dossier data avec les images contenu dans le fichier img_align_celeba

NB_IMAGES =202599


def process_images():
    
    # On regarde si les images n'ont pas déjà été processsed
    
    # if os.path.isfile("data/processed_img"):
    #     print("It seems that images have already been processed : data/processed_img")
    #     return
    # else:
    #     os.mkdir("data/processed_img")

    #Take images in /data/img_aligned_celeba, processes it and save it in /data/processed_img
    print("Redimensionnemment des images : ")
    img_tab =[]
    count = 1
    for i in range(1, NB_IMAGES+1):
        if i%1000 == 0:
            print(i, "/", NB_IMAGES)
        img = cv.imread("data/img_align_celeba/%06i.jpg" %i)
        img = cv.resize(img, (256,256), interpolation=cv.INTER_LANCZOS4)
        img_tab.append(img)
        if i % 20000 == 0:
            np.savez_compressed(f"data/img_celeba{count}.npz", img_tab)
            count +=1
            img_tab = []




    print("Images have been processed")


def process_attributes():
    # Les attributs sont stocké dans un fichier list_attr_celeba.txt
    i = 0
    attr = []
    for line in open('data/list_attr_celeba.txt', 'r'):
        attr.append(line.rstrip()) # Une sécurité
    
    # La premlière ligne du fichier contient le nombre d'attributs
    # La deuxieme contient les catégorie d'attributs (sourire, moustache, lunette...)
    keys = attr[1].split() 
    dict_attr = {}
    for k in keys:
        dict_attr[k] = np.zeros(NB_IMAGES)
    
    for i, line in enumerate(attr[2:]): 
        split = line.split()
        assert len(split) == len(keys) + 1

        for k, l in enumerate(split[1:]):
            # Comme écrit dans l'article, au lieu d'utiliser 1 et -1, on utilisera des valeurs binaire 1 et 0
            dict_attr[keys[k]][i] = l == '1' 

    #dict_attr est un dictionnaire où pour chaque clé (sourire, lunette ...) correspond un tableau de taille NB_IMAGES
    np.savez_compressed("data/attributes.npz",dict_attr)
    return dict_attr



dicts = process_attributes()
print(dicts["Attactive"])
# process_images()


