from BiT import bio_taxo
import mahotas.features as ft
from skimage.feature import graycomatrix, graycoprops # scikit-image
import numpy as np

# Define Bitdesc
def bitdesc(data):
    all_statistics = bio_taxo(data)
    return all_statistics

def bitdesc_glcm(data):
    # Extract features from Bitdesc and GLCM
    bitdesc_features = bitdesc(data)
    glcm_features = glcm(data)
    
    # Concatenate the features
    concatenated_features = bitdesc_features + glcm_features
    
    return concatenated_features

def bitdesc_haralick(data):
    # Extract features from Bitdesc and Haralick
    bitdesc_features = bitdesc(data)
    haralick_features = haralick_with_mean(data)  # Utilize haralick_with_mean function to get mean values
    
    # Concatenate the features
    concatenated_features = bitdesc_features + haralick_features
    
    return concatenated_features
# Define Haralick
def haralick(data):
    all_statistics = ft.haralick(data).astype("float").tolist()
    return all_statistics
def haralick_with_mean(data):
    all_statistics = ft.haralick(data).mean(0).astype("float").tolist()
    return all_statistics

# Gray-Level Co-occurence Matrix
def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]    
    all_statistics = [diss, cont, corr, ener, homo]
    return all_statistics







