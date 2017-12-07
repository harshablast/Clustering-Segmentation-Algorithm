# Clustering-Segmentation-Algorithm
An alternative algorithm for segmentation in images based on clustering of regional based histograms.

This project aims to introduce a new method of segmentation in digital images which utilizes clustering as the method of differentiation of super pixels in the image. This algorithm takes into account the features of neighbouring pixels of a particular pixel prior to clustering and assigning it to a segment.

Segmentation techniques use features to differentiate between the super pixels of the image. Common segmentation features include colour information, texture information, gradients, location, etc. Our algorithm utilizes colour, texture and gradient information of neighbouring pixels. 



While most segmentation algorithms have a global approach and considers the features of the whole image, our algorithm does not completely follow this method. We divide the image into many parts and get the features of each part of the image. These features then get clustered and then based on the features of neighbouring pixels of a pixel in the image, it will get segmented. 
While this algorithm can take a lot of time to run due to the iteration and feature calculation of each and every pixel and its neighbouring pixels in the image, but it allows us to apply region based classification techniques like texture/colour classification and then use them to cluster the image and segment it at a higher accuracy. 

In our algorithm, we first resize the image to some standard size and divide the image into several rectangles in a grid-like format. The following features are then calculated for each rectangle: 
I. LBP histogram II. RGB histograms III. HOGS 
 
These histograms are then appended together and added to a cumulative list of features, each rectangle’s features being separate. We then set a number of clusters, based on the image we are segmenting, and then proceed to perform Kmeans clustering on the list of features.  

When we do this, it essentially allows us to cluster the image based on neighbouring pixel features like LBP and HOGS and since these classification techniques require a regional histogram, we divide the image into rectangles and get their features. Essentially we are ‘classifying’ the rectangles of the image based of the features and assigning them to a cluster. 
After this, we iterate through each and every pixel of the image, for each pixel, the following occurs: 
I. A temporary region of interest in considered with the same size of the rectangles in the previous part of the algorithm, the region having the centre as the target pixel. II. The LBP, RGB and HOGS histograms/features are collected from this region of interest and appended in exactly the same format as the previous part of the algorithm. III. This feature vector of the region of interest is then input into a prediction function which returns the cluster it belongs to. IV. The target pixel is then given a value corresponding to the cluster it’s region of interest belongs to. 
This is repeated for each and every pixel and hence every pixel will be assigned to a cluster and the image is henceforth segmented.  
The hyper parameters of the algorithm include: 
a) Window size for gathering feature vectors. b) LBP parameters: a. Number of points b. Radius c) Number of clusters 
While making our algorithm, we used window size of 25 pixels height and width and found it yielded the best results as it is big enough to capture important information and small enough to not divide the image too less. The LBP parameters we used were 8 points and radius 3, these could change to give better results for different type of images. The number of clusters was something we had to select based on the input image. 
