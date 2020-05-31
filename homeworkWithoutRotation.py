import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img1 = cv2.imread('1-1.jpg')
img2 = cv2.imread('1-2.jpg')

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

dx, dy = 0, 0
best_error = 200


# 平行移動
for y in range(30, 50):
    for x in range(150, 250):
        error = 0
        count = 0
        normalized_error = 0
        for i in range(60):
            for j in range(60):
                if (i<0 or i>=h1 or j<0 or j>=w1):
                    continue
                count += 1
                error += np.linalg.norm(img1[y+i, x+j, :] - img2[i, j, :], ord=2)
        if count==0:
            continue
        else:
            normalized_error = error / count
        
        if normalized_error<best_error:
            best_error = normalized_error
            dx = x
            dy = y
            print(best_error, dx, dy)
    print(y)

print(f'dx:{dx}, dy:{dy}, best_error:{best_error}')

# 出力
img = np.zeros((h1+dy, w1+dx, 3), dtype=np.uint8)

img[0:h1, 0:w1, :] = img1

img[dy:h2+dy, dx:w2+dx, :] = img2

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()

save_img = img

cv2.imwrite("save_img1_dx_{}_dy_{}.jpg".format(dx, dy),save_img)