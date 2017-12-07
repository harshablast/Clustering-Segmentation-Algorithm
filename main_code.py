import cv2
import numpy as np
from skimage import feature
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('image11.jpg',0)
img = cv2.resize(img,(1920,1080),interpolation=cv2.INTER_LANCZOS4)
img_colour = cv2.imread('image11.jpg',1)
img_colour = cv2.resize(img_colour,(1920,1080),interpolation=cv2.INTER_LANCZOS4)

segmented = img.copy()

box_width = 24
box_height = 27

image_list = []
lbp_hist = []

#edges = cv2.Canny(img,100,200)
#im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,0), 1)

lbp = feature.local_binary_pattern(img,8,3)
lbp=lbp.astype(np.uint8)

for i in range(80):
    for j in range(40):
        roi_lbp=lbp[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width]
        roi_colour=img_colour[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width]
        image_list.append(roi_lbp)
        temp_hist_lbp = cv2.calcHist(roi_lbp,[0],None,[256],[0,256])

        temp_hist_colour_b = cv2.calcHist(roi_colour,[0],None,[256],[0,256])
        temp_hist_colour_g = cv2.calcHist(roi_colour, [1], None, [256], [0, 256])
        temp_hist_colour_r = cv2.calcHist(roi_colour, [2], None, [256], [0, 256])

        #total_hist=temp_hist_lbp*3+temp_hist_colour_b+temp_hist_colour_g+temp_hist_colour_r
        total_hist = np.concatenate((temp_hist_lbp, temp_hist_colour_b, temp_hist_colour_g, temp_hist_colour_r), axis=0)
        lbp_hist.append(total_hist)



lbp_hist=np.array(lbp_hist)
lbp_hist=np.squeeze(lbp_hist,2)

clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(lbp_hist)
print kmeans.labels_

for i in range(80):
    for j in range(40):
        temp_index=(i*40)+j
        temp_label=kmeans.labels_[temp_index]
        segmented[j * box_height:(j + 1) * box_height, i * box_width:(i + 1) * box_width]=(256//clusters)*temp_label


#x = itemfreq(lbp.ravel())
#Normalize the histogram
#hist = x[:, 1]/sum(x[:, 1])
#plt.plot(x)
#plt.title("lol")
#plt.show()

img=cv2.resize(img,(1000,500))
img_colour=cv2.resize(img_colour,(1000,500))
segmented=cv2.resize(segmented,(1000,500))
lbp=cv2.resize(lbp,(1000,500))


cv2.imshow("Original",img_colour)
cv2.imshow("Segmented",segmented)
cv2.imshow('LBP',lbp)

cv2.waitKey(0)
cv2.destroyAllWindows()