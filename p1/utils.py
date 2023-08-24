import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist
# import cv2 #new
# from time import time
from scipy.stats import mode

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths):
    '''
    Input : 
        img_paths (N) : list of string of image paths
    Output :
        tiny_img_feats (N, d) : ndarray of resized and then vectorized 
                                tiny images
    NOTE :
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image, making the     #
    #       tiny images unit length and zero mean, which will       #
    #       slightly increase the performance                       #
    #################################################################

    tiny_img_feats = []
    #read img from path(1500 img)
    for img_path in img_paths:
        img = Image.open(img_path)
        tiny_img = img.resize((16, 16)) # resize
        tiny_img = np.array(tiny_img).flatten() # flatten
        tiny_img = (tiny_img - np.mean(tiny_img)) / np.std(tiny_img) #normalization (x-mean) / SD
        tiny_img_feats.append(tiny_img)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(img_paths, vocab_size=500):
    '''
    Input : 
        img_paths (N) : list of string of image paths (training)
        vocab_size : number of clusters desired
    Output :
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    NOTE :
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better (to a point) 
           but be slower to compute, you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
   
    # create a list to collect features
    feats = []
    
    # loop through all the image paths
    for img_path in tqdm(img_paths): ### tqdm
        # load the image
        img = np.asarray(Image.open(img_path),dtype='float32')
        # get its 128-dim SIFT features
        kp, descs = dsift(img, step=[3, 3], fast=True)
        # append the features to the list
        feats.append(descs)
    
    # stack all the features vertically to form a numpy array
    feats = np.vstack(feats).astype('float32') # stack multiple arrays in a list along the vertical direction (i.e., along the row direction)
    
    ### time 
    print("start building vocab")  

    # perform k-means clustering on the features
    # vocab = kmeans(feats, num_centers=vocab_size, verbose=False)[0]
    vocab = kmeans(feats, vocab_size, initialization="PLUSPLUS")  
    # print(vocab.shape) # 400*128
    # print(vocab.dtype) # float32
    # print(vocab.ndim) # 2

    print("finish building vocab")

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab
    # return None

###### Step 1-b-2
def get_bags_of_sifts(img_paths, vocab):
    '''
    Input :
        img_paths (N) : list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Output :
        img_feats (N, d) : ndarray of feature of images, each row represent
                           a feature of an image, which is a normalized histogram
                           of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []

    ### time
    # start_time = time()
    print("starting bags of sifts")

    for img_path in tqdm(img_paths):
        # Load image
        img = Image.open(img_path)

        # Convert image to numpy array
        img = np.array(img)

        # Extract SIFT features
        step_size = 4
        sift_features, desc = dsift(img, step=[step_size, step_size], fast=True)

        # Calculate distances between features and cluster centers
        distances = cdist(vocab, desc) # chose metric

        # Assign each feature to nearest cluster center
        assignments = np.argmin(distances, axis=0) # axis = 0

        # Build histogram of cluster assignments
        # vocab is a set of cluster centers, its shape is (cluster_centers, feature_dim), so vocab.shape[0] is equal to the number of cluster_centers
        histogram, _ = np.histogram(assignments, bins=vocab.shape[0], density=True)

        # Normalize histogram
        histogram = [float(i)/sum(histogram) for i in histogram]

        # Append histogram to list of image features
        img_feats.append(histogram)

    # Convert list of histograms to numpy array
    img_feats = np.array(img_feats)

    # end_time = time()
    print("finish bags of sifts")

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(train_img_feats, train_labels, test_img_feats):
    '''
    Input : 
        train_img_feats (N, d) : ndarray of feature of training images
        train_labels (N) : list of string of ground truth category for each 
                           training image
        test_img_feats (M, d) : ndarray of feature of testing images
    Output :
        test_predicts (M) : list of string of predict category for each 
                            testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
           'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
           'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    CAT2ID = {v: k for k, v in enumerate(CAT)}

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []

    K = 5 # choice K value (num for compare)

    # first compare the train and test img feats distance
    '''
    cdist.(a, b, )
    a, b: 大小為(m,d) - 成員數/每個成員的維度的兩個集合
    metric = distance calculate method
    output: size = (n,m)的二維矩陣，其中每個元素都是兩個集合中成員之間的距離
    '''
    dist = cdist(test_img_feats, train_img_feats, metric='minkowski', p=0.5) # metric
    # metric='minkowski', p=2


    # find K nearest training labels for each test image
    k_nearest_labels = np.array(train_labels)[np.argsort(dist)[:,:K]]
    
    # most common label wins ! # not worlking
    # test_predicts = mode(k_nearest_labels,axis=1).mode.ravel()

    # Vote for the final label using the k-nearest labels: this work
    for i in range(k_nearest_labels.shape[0]):
        nearest_labels = k_nearest_labels[i]
        nearest_labels_id = [CAT2ID[label] for label in nearest_labels]
        final_label_id = mode(nearest_labels_id)[0][0]
        final_label = CAT[final_label_id]
        test_predicts.append(final_label)

    # # sort the dist from low to high (num = K)
    # knn_idxs = np.argsort(dist, axis=1)[:, :K] # 1500*K
    # # knn_idxs is a matrix of the shape (M, K), where M is the number of test sets and K is the number of chosen nearest neighbors

    # train_labels = np.array([int(label) for label in train_labels])

    # # Find the closest K training images to which the category labels then used to predict the category of the test image
    # # add most common label(vote)
    # for i in range(knn_idxs.shape[0]): # 1500 runs
    #     knn_labels = train_labels[knn_idxs[i]]
    #     # Initializes a numpy array of votes with all elements 0, of size = number of all possible classes
    #     votes = np.zeros(len(CAT)) 
    #     # For each of these K neighboring tags, find its corresponding index and add 1 to the corresponding element in the votes array
    #     for label in knn_labels:
    #         votes[CAT2ID[label]] += 1
    #     # index of the maximum value in votes, and find its corresponding class
    #     test_predicts.append(CAT[np.argmax(votes)])

    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
