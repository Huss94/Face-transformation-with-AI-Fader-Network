import numpy as np 
import matplotlib.pyplot as plt

history = np.load("models/Young/history.npy", allow_pickle = True).item()

plt.figure()
for p in history:
    # if p == 'dis_accuracy' or p== 'dis_val_accuracy':
    plt.plot(np.arange(0,len(history[p]),1), history[p], label = p)

plt.legend()
plt.show()


