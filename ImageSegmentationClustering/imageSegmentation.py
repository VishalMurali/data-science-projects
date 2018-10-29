# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:37:49 2016
Image Segmentation
@author: Win
"""
from PIL import Image
import numpy as np
import sys

'''
This method reassigns the clusers
'''

def newCluster(X, K, label):
	# return K-by-5 new centroids
	new = np.matrix(np.zeros((K,5)))

	for i in range(K):
		index = np.array(np.where(label==i)[0]).squeeze()
		Xsub = X[index, :]
		new[i,:] = np.mean(Xsub, axis=0)

	return new
'''

This method returns the average sistance of the centers of K clusters
'''
def avg(c1, c2):
	
      
	diff = c1 - c2
	norm = np.sum(np.multiply(diff,diff), axis=1)
	return np.mean(np.sqrt(norm))

'''
This method returns an n-by-K matrix, distance between each instance and K centroids
'''

def distance_matrix(X, K, centroid):
	# 
	n,d = X.shape
	distance = np.matrix(np.zeros((n,K)))

	for i in range(K):
		diff = X - centroid[i,:]
		distance[:,i] = np.sum(np.multiply(diff,diff), axis=1)

	return distance

    
if __name__ == '__main__':
    
    """
    Taking in arguments from the command line
    """
    epsilon = 0.001
    outputImageFilename = str(sys.argv[-1])
    inputImageFilename = str(sys.argv[-2]) 
    img = Image.open(inputImageFilename)
    K = np.uint8(sys.argv[-3])
    
    #img = Image.open("C:\Users\Win\Desktop\Coursework\CIS 519 - Machine Learning\HW5\singapore1.jpg")
    '''
    Converting the image into rgb pixels, to a matrix containing xp = [rp gp bp ip jp]
    '''
    xres, yres = img.size
    # Get raw data. xp=[Rp,Gp,Bp,ip,jp]
    X = []
    for j in range(yres):
	 for i in range(xres):
			r,g,b = img.getpixel((i,j))
			X.append([r,g,b,i,j])
    X = np.matrix(X)
    
    '''
    Standardizing the matrix obtained above
    '''
    
    n,d = X.shape
    X_mean = np.mean(X, axis=0)  
    X_std = np.std(X, axis=0)   
    X = (X - X_mean) / X_std
    
    # Pick initial centroid.
    w = np.array([0.2, 0.2, 0.2, 0.25, 0.25])
    score = X.dot(w)
    index1 = score.argsort(axis=1)  
    index2 = np.int_(np.array(range(1,K+1))/float(K)*n-1)  
    index = index1[0,index2]
    
    centr_now = X[index, :].squeeze()
    centr_pre = np.zeros((K,5))
    	
    
    print 'Assigning pixels to clusters : '
    i = 1
    
    while avg(centr_now,centr_pre) > epsilon:
    		print 'Iteration: ', i
    		centr_pre = centr_now
    		kdist = distance_matrix(X, K, centr_now)
    		label = np.argmin(kdist, axis=1)
    		centr_now = newCluster(X, K, label)
    		i += 1
    		if i >