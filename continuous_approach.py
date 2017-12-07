import cv2
import numpy as np
from skimage import feature
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('image11.jpg',0)
img = cv2.resize(img,(2000,1000),interpolation=cv2.INTER_LANCZOS4)
img_colour = cv2.imread('image11.jpg',1)
img_colour = cv2.resize(img_colour,(2000,1000),interpolation=cv2.INTER_LANCZOS4)

segmented = img.copy()

box_width = 25
box_height = 25

lbp = feature.local_binary_pattern(img,8,3)
lbp=lbp.astype(np.uint8)

feature_hist = []

for i in range(80):
    for j in range(40):
        roi_lbp=lbp[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width]
        roi_colour=img_colour[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width]

        temp_hist_lbp = cv2.calcHist(roi_lbp,[0],None,[256],[0,256])

        temp_hist_colour_b = cv2.calcHist(roi_colour,[0],None,[256],[0,256])
        temp_hist_colour_g = cv2.calcHist(roi_colour, [1], None, [256], [0, 256])
        temp_hist_colour_r = cv2.calcHist(roi_colour, [2], None, [256], [0, 256])

        total_hist=np.concatenate((temp_hist_lbp,temp_hist_colour_b,temp_hist_colour_g,temp_hist_colour_r), axis=0)
        feature_hist.append(total_hist)

total_hist= np.array(total_hist)
print total_hist
print len(total_hist)
print total_hist.shape
total_hist= np.squeeze(total_hist,1)
print total_hist
feature_hist=np.array(feature_hist)
feature_hist=np.squeeze(feature_hist,2)

clusters = 4
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(feature_hist)
print kmeans.labels_

predictions = []

for i in range(13,1988):
    for j in range(13,988):
        roi_lbp = lbp[j-12:j+12,i-12:i+12]
        roi_colour = img_colour[j-12:j+12,i-12:i+12]

        temp_hist_lbp = cv2.calcHist(roi_lbp, [0], None, [256], [0, 256])

        temp_hist_colour_b = cv2.calcHist(roi_colour, [0], None, [256], [0, 256])
        temp_hist_colour_g = cv2.calcHist(roi_colour, [1], None, [256], [0, 256])
        temp_hist_colour_r = cv2.calcHist(roi_colour, [2], None, [256], [0, 256])

        total_hist = np.concatenate((temp_hist_lbp, temp_hist_colour_b, temp_hist_colour_g, temp_hist_colour_r), axis=0)
        #total_hist = temp_hist_lbp * 3 + temp_hist_colour_b + temp_hist_colour_g + temp_hist_colour_r
        total_hist=np.squeeze(total_hist,1)
        prediction = kmeans.predict([total_hist])
        predictions.append(prediction)
        segmented.itemset((j,i),256//clusters*prediction[0])
        print i*975 + j

print predictions
print (total_hist)

segmented=cv2.resize(segmented,(1000,500))
cv2.imshow('CLustered',segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()
