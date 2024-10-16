import numpy as np

def generate_shape(label, size=16):
    img = np.zeros((size, size))
    
    if label == 0:
        img[4:12, 4:12] = 1
    elif label == 1:
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//4)**2
        img[mask] = 1
    elif label == 2:
        img[4:12, 4:12] = 1
        for i in range(6):
            img[i, 4 + i] = 0  
    elif label == 3:
        np.fill_diagonal(img, 1)
        np.fill_diagonal(np.fliplr(img), 1)
    elif label == 4:
        img[7:9, :] = 1
        img[:, 7:9] = 1
    
    return img

