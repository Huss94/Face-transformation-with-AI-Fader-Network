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
            np.savez_compressed(f"data/img_celebacomp{count}.npz", img_tab)
            count +=1
            img_tab = []




    print("Images have been processed")


process_images()


