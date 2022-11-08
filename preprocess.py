import os 
import cv2 as cv
import numpy as np
import argparse


# Il faut que l'utilisateur ait le dossier data avec les images contenu dans le fichier img_align_celeba

NB_IMAGES =202599
parser = argparse.ArgumentParser('preprocessing data')
parser.add_argument("--img_path", type= str, default = "data/img_align_celeba")
parser.add_argument("--img_save_path", type= str, default = "data/img_align_celeba_resized")

parser.add_argument("--attr_path", type= str, default = "data/list_attr_celeba.txt")
parser.add_argument("--attr_save_path", type= str, default = "data/attributes.npz")

params = parser.parse_args()
def process_images():
    
    # On regarde si les images n'ont pas déjà été processsed
    
    if os.path.isdir(params.img_save_path):
        print("It seems that images have already been processed : " + params.img_save_path)
        return
    else:
        os.mkdir(params.img_save_path)

    #Take images in /data/img_aligned_celeba, processes it and save it in /data/processed_img
    print("Redimensionnemment des images : ")
    img_tab =[]
    # count = 1
    for i in range(1, NB_IMAGES+1):
        if i%1000 == 0:
            print(i, "/", NB_IMAGES)
        img = cv.imread(params.img_path + "/%06i.jpg" %i)
        img = cv.resize(img, (256,256), interpolation=cv.INTER_LANCZOS4)
        # img_tab.append(img)
        cv.imwrite(params.img_save_path + "/%06i.jpg" %i, img) 
    
    print("Images have been processed")


def process_attributes():
    # Les attributs sont stocké dans un fichier list_attr_celeba.txt

    if os.path.isfile(params.attr_save_path):
        print("It seems that attributes have already been processed : " + params.attr_save_path)
        return
    i = 0
    attr = []

    for line in open(params.attr_path, 'r'):
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
            # Comme écrit dans l'article, au lieu d'utiliser 1 et -1, on utilisera des valeurs binaire 1 et 0
            dict_attr[keys[k]][i] = l == '1' 

    #dict_attr est un dictionnaire où pour chaque clé (sourire, lunette ...) correspond un tableau de taille NB_IMAGES
    np.savez_compressed("data/attributes.npz",dict_attr)
    return dict_attr





process_images()
process_attributes()


