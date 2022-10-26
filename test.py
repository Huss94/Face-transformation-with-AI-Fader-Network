import numpy as np 

# a = np.load("data/img_celeba1.npz")['arr_0']
b = np.load("data/attributes.npz", allow_pickle=True)['arr_0'].item() #Il faut faire item poru réccuperer le dictionnaire
# print(a[0])
print(b)

input("ok")
