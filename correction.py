import numpy as np
seed=42
np.random.seed(seed)
def spatialCorrection(Y,r=1): #Y is arranges in a grid

    Y_padded = np.pad(Y ,r, mode='constant')

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            s, t = 2 * r + i + 1, 2 * r + j + 1
            window = Y_padded[i:s, j:t]
            if(Y[i][j])==0: ##correggo solo i normali
                if (np.count_nonzero(window == 1) > np.count_nonzero(window == 0)):
                    Y[i][j] = 1
                elif (np.count_nonzero(window == 0) > np.count_nonzero(window == 1)):
                    Y[i][j] = 0

    return Y
