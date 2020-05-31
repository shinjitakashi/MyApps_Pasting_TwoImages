import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# image1-1, image1-2を合成
img1 = cv2.imread('1-1.jpg')
img2 = cv2.imread('1-2.jpg')

# image2-1, image2-2を合成
# img1 = cv2.imread('2-1.jpg')
# img2 = cv2.imread('2-2.jpg')


h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

dx, dy, da = 0, 0, 0
best_error = 200

for angle in range(0, 20, 5):
    s1 = math.sin(angle * math.pi / 180)
    c1 = math.cos(angle * math.pi / 180)
    # 平行移動
    for y in range(20, 60, 2):
        for x in range(160, 240, 2):
            error = 0
            count = 0
            normalized_error = 0
            for i in range(20):
                for j in range(20):
                    v = math.floor(s1 * i + c1 * j)
                    u = math.floor(c1 * i - s1 * j)
                    if (y+v<0 or y+v>=h1 or x+u<0 or x+u>=w1):
                        continue
                    count += 1
                    error += np.linalg.norm(img1[y+v, x+u, :] - img2[v, u, :], ord=2)
            if count==0:
                continue
            else:
                normalized_error = error / count
            
            if normalized_error<best_error:
                best_error = normalized_error
                dx = x
                dy = y
                da = angle
                print(best_error, dx, dy, da)

print(f'dx:{dx}, dy:{dy}, da:{da} best_error:{best_error}')

# 出力
def rotate_matrix(theta, x_t, y_t):
    A = np.array([[np.cos(math.radians(theta)), -np.sin(math.radians(theta)), x_t],
             [np.sin(np.math.radians(theta)), np.cos(np.math.radians(theta)), y_t],
             [0,0,1]])
    return np.linalg.inv(A)
    
def afin_image(A, image, x_length, y_length):
    afin_result = np.zeros((y_length, x_length, 3), dtype=np.uint8)
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x].any() == 0:
                continue
            trans_posi = A @ np.array([x, y, 1])
            index_x = math.floor(trans_posi[0])
            index_y = math.floor(trans_posi[1])
            if index_x >= 0 and index_x < x_length and index_y >= 0 and index_y < y_length:
                afin_result[y][x] = image[index_y][index_x]
    return afin_result

# image1-1, image1-2を合成
h, w = img1.shape[:2]
img = np.zeros((h+80, w+250, 3), dtype=np.uint8)

inv_A = rotate_matrix(da, 0, 0)
afin_result = afin_image(inv_A, img2, w, h)

img = np.zeros((h1+dy, w1+dx, 3), dtype=np.uint8)

img[dy:h2+dy, dx:w2+dx, :] = afin_result

img[0:h1, 0:w1, :] = img1

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()

save_img = img

cv2.imwrite("save_img1_dx_{}_dy_{}.jpg".format(dx, dy),save_img)

# image2-1, image2-2を合成
# h, w = img1.shape[:2]
# img = np.zeros((h+80, w+250, 3), dtype=np.uint8)

# inv_A = rotate_matrix(da, 0, 0)
# afin_result = afin_image(inv_A, img2, w, h)

# img = np.zeros((h1+dy, w1+dx, 3), dtype=np.uint8)

# img[dy:h2+dy, dx:w2+dx, :] = afin_result

# img[0:h1, 0:w1, :] = img1

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# plt.show()

# save_img = img

# cv2.imwrite("save_img2_dx_{}_dy_{}.jpg".format(dx, dy),save_img)