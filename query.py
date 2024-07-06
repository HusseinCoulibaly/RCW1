import cv2
from descripteurs import bitdesc
import numpy as np
from scipy.spatial import distance

def main():
    query = 'test.png'
    # Load Signatures
    signatures = np.load('cbir_signatures_v1.npy')
    print(signatures.shape)
    # List of distances
    distanceList = list()
    img = cv2.imread(query, 0) # 0 means gray-level
    # Extract signatures from query image
    bit_feat = bitdesc(img)
    # Compute and store distances
    for sign in signatures:
        # REmove the last two element of the array
        sign = np.array(sign)[0:-2].astype('float')
        # Convert numpy array to list
        sign = sign.tolist()
        # Compute distance (cityblock-Manhattan)
        dist = distance.euclidean(bit_feat, sign)
        distanceList.append(dist)
        # print(f'Distance: {dist}')
    print('Distance computed Successfully!')
    # Grab indices of min distances
    minDistances = list()
    # Example: Display 10 most similar images
    for i in range(10):
        array = np.array(distanceList)
        # Pick the index of min distance
        min_Element_index = np.argmin(array)
        # Add min index to list of min distances
        minDistances.append(min_Element_index)
        # distanceList[min_Element_index] = np.inf
        distanceList[min_Element_index] = array.max()
        
    print(f'Index of 10 most similar signatures: {minDistances}')
    # [331, 1077, 1060, 1084, 947, 1086, 1116, 1072, 1079, 1062]
    # Display signatures of n most similar images
    
    for small in minDistances:
        print(signatures[small][-2:])
    #     img = cv2.imread(signatures[small][-1])
    #     cv2.imshow(f'Image Index: {small}',img)
    # cv2.waitKey(0)
        
        

if __name__ == '__main__':
    main()
    
    