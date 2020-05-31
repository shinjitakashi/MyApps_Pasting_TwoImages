import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

kernel = [[1/256, 4/256, 6/256, 4/256, 1/256],
[4/256, 16/256, 24/256, 16/256, 4/256], [6/256, 24/256, 36/256, 24/256, 6/256], [4/256, 16/256, 24/256, 16/256, 4/256], [1/256, 4/256, 6/256, 4/256, 1/256]]

# for image1-1 image1-2
img1 = cv2.imread('1-1.jpg')
img2 = cv2.imread('1-2.jpg')

# # for image2-1 image2-2
# img1 = cv2.imread('2-1.jpg')
# img2 = cv2.imread('2-2.jpg')

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

img1_gray1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
img2_gray1 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]


def differential(s, h, w):
    sx = np.zeros((h, w), dtype=s.dtype)
    sy = np.zeros((h, w), dtype=s.dtype)
    for y in range(h - 1) :
        for x in range(w - 1) :
            sx[y, x] = s[y, x + 1] - s[y, x]
            sy[y, x] = s[y + 1, x] - s[y, x]
    return sx, sy

def Harris(s1, s2, s3, kernel):
    harris_result = np.zeros(s1.shape, dtype=s1.dtype)
    h, w = s1.shape
    for y in range(2 ,h - 2, 2) :
        for x in range(2 , w - 2, 2) :
            l = 0
            m = 0
            n = 0
            for u in range(-2 , 3) :
                for v in range(-2 , 3) :
                    l += s1[y + u, x + v] * kernel[u + 2][v + 2]
                    m += s2[y + u, x + v] * kernel[u + 2][v + 2]
                    n += s3[y + u, x + v] * kernel[u + 2][v + 2]
            c = np.array([[l, m], [m, n]])
            k1, k2 = np.linalg.eig(c)
            # k = 0.05
            harris_result[y, x] = k1[0]*k1[1] - 0.05*(k1[0] + k1[1])
    return harris_result

def point_dx_dy(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    dx, dy, da = 0, 0, 0
    best_error = 20000
    for angle in range(0, 30, 5):
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
                        error += np.linalg.norm(img1[y+v, x+u] - img2[v, u])
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
    
    return dx, dy, da, best_error

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



f1x, f1y = differential(img1_gray1, h1, w1)
f2x, f2y = differential(img2_gray1, h2, w2)

s1 = f1x * f1x
s2 = f1x * f1y
s3 = f1y * f1y
s4 = f2x * f2x
s5 = f2x * f2y
s6 = f2y * f2y
s_img1 = Harris(s1, s2 , s3, kernel)
s_img2 = Harris(s4, s5 , s6, kernel)

plt.imshow(s_img1)
plt.savefig('s_1-1.jpg')

plt.imshow(s_img2)
plt.savefig('s_1-2.jpg')

cv2.imwrite("s_image1-1.jpg",s_img1)
cv2.imwrite("s_image1-2.jpg",s_img2)

"""
ここまでがHarrisの実装
"""
"""
dx, dy, da, best_error = point_dx_dy(s_img1, s_img2)

h, w = img1.shape[:2]
img = np.zeros((h+80, w+250, 3), dtype=np.uint8)

inv_A = rotate_matrix(da, 0, 0)
afin_result = afin_image(inv_A, img2, w, h)

img = np.zeros((h1+dy, w1+dx, 3), dtype=np.uint8)

img[dy:h2+dy, dx:w2+dx, :] = afin_result

img[0:h1, 0:w1, :] = img1

print(f'dx:{dx}, dy:{dy}, da:{da} best_error:{best_error}')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()

save_img = img

"""
