import numpy as np

def generate_shape(label, size=16):
    img = np.zeros((size, size))
    
    if label == 0:
        # Square
        img[4:12, 4:12] = 1
    elif label == 1:
        # Circle
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//4)**2
        img[mask] = 1
    elif label == 2:
        # Triangle
        for i in range(8): 
            img[8 - i, 8 - i:8 + i + 1] = 1  
    elif label == 3:
        # X pattern:
        np.fill_diagonal(img, 1)
        np.fill_diagonal(np.fliplr(img), 1)
    elif label == 4:
        # Plus
        img[7:9, :] = 1
        img[:, 7:9] = 1
    
    return img