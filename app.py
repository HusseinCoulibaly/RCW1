import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from descripteurs import bitdesc, glcm, haralick_with_mean,bitdesc_glcm,bitdesc_haralick
import pandas as pd
from distances import distance_selection
from upload import upload_file

def main():
    def training(X,Y):
        model = LogisticRegression()
        model.fit(X, Y)
        return model
    st.write("App lancée!")
    st.write('''
        Projet2 😀
    ''')
     # Entrer le nombre  de resultats
    input_value = st.sidebar.number_input("Entrez une valeur", min_value=1, max_value=500, value=10, step=1)
    st.sidebar.write(f"Vous avez entré {input_value}")
    # Define distances
    col1, col2 = st.sidebar.columns(2)

    # Definition des distances
    choix = ["Euclidean", "Canberra", "Manhattan", "Chebyshev", "Minkowsky"]
    choix_distances = col1.radio("Selectionnez une distance", choix)
    st.write(f"Vous avez choisi {choix_distances}")

    # Definition des descripteurs 
    desc = ["GLCM","BITDESC","HARALICK","BITDESC+GLCM","BITDESC+HARALICK"]
    choix_descripteur = col2.radio("Selectionnez un descripteur", desc)
    st.write(f"Vous avez choisi {choix_descripteur}")
    # declaration de  X et Y
    X = None
    Y = None
    # Import hors connexion de database (signatures)
    if choix_descripteur == "GLCM":
        signatures = np.load('cbir_signatures_glcm.npy')
    elif choix_descripteur == "BITDESC":
        signatures = np.load('cbir_signatures_bitdesc.npy')
    elif choix_descripteur == "HARALICK":
        signatures = np.load('cbir_signatures_haralick_with_mean.npy')
    elif choix_descripteur=="BITDESC+GLCM":
        signatures = np.load('cbir_signatures_bitdesc_glcm.npy')
    elif choix_descripteur == "BITDESC+HARALICK":
        signatures = np.load("cbir_signatures_bitdesc_haralick.npy")

  
    # Define a list for computed distances
    distanceList = list()
    # Upload image
    is_image_uploaded = upload_file()
    if is_image_uploaded:
        start_time = time.time()
        st.write('''
                 # Search Results
                 ''')
        # Retrieve query image
        query_image = 'uploaded_images/query_image.png'
        # lire l'image en gray-scale
        img = cv2.imread(query_image, 0)
        # Get signatures (extract features) of query image/Compute Bitdesc
        if choix_descripteur == "GLCM":
            bit_feat = glcm(img)
        elif choix_descripteur == "BITDESC":
            bit_feat = bitdesc(img)
        elif choix_descripteur == "HARALICK":
            bit_feat = haralick_with_mean(img)
        elif choix_descripteur == "BITDESC+GLCM":
            bit_feat = bitdesc_glcm(img)
        elif choix_descripteur == "BITDESC+HARALICK":
            bit_feat = bitdesc_haralick(img)
        
        if signatures is not None:
            X = signatures[:, :-2]
            Y = signatures[:, -2]
            model = training(X,Y)
        # convertir bit_feat en numpy array
        conv = np.array(bit_feat)
        matrice2d = conv.reshape(1, -1)
        prediction = model.predict(matrice2d)
        # Compute Similarity distance
        for sign in signatures:
            # Remove the last two columns ('subfolder', 'path')
            sign = np.array(sign)[0:-2].astype('float')
            # Convert numpy to list
            sign = sign.tolist()
            # Call distance function
            distance = distance_selection(choix_distances, bit_feat, sign)
            distanceList.append(distance)
        
        print("Ditances computed successfully")
        # Compute n min distances
        minDistances =list()
        for i in range(input_value):
            array = np.array(distanceList)
          
            index_min = np.argmin(array)
            minDistances.append(index_min)
            max = array.max()
           
            distanceList[index_min] = max
        print(minDistances)
 
        image_paths = [signatures[small][-1] for small in minDistances]
        # Retrieve classes/Types of most similar images using their distances
        classes = [signatures[small][-2] for small in minDistances]
        classes_ = np.array(classes)
        # Get unique values of types and count all
        unique_values, counts = np.unique(classes_, return_counts=True)
        list_classes = list()
        print("Unique value with their counts")
        for value, count in zip(unique_values, counts):
            print(f"{value}:{count}")
            list_classes.append(value)
        # Create pandas Dataframe with the unique value and their counts
        df = pd.DataFrame({"Value": unique_values, "frequency":counts})
       
        # division en deux colonnes de 75% 25%
        col1, col2 = st.columns([3, 1])
        with col1:
            # Plot bar chart and set Value as index. Frequency as value to display
            st.bar_chart(df.set_index("Value"))
        with col2:
            # Temps d'execution
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"PREDICTION : {prediction[0]} ")
            st.write(f"TIME : {elapsed_time:.4f} secondes")
        # Display
        col1, col2, col3 = st.columns(3)
        for i in range(input_value):
            image = Image.open(image_paths[i])
            # Display the upload image with Streamlit
            if i % 3 == 0:
                col1.image(image, caption=classes[i], use_column_width=True)
            elif i % 3 == 1:
                col2.image(image, caption=classes[i], use_column_width=True)
            else:
                col3.image(image, caption=classes[i], use_column_width=True)

    else:
        st.write("Bienvenue! s'il vous plait  veuillez charger une image pour pouvoir commencer ...")
                                   
if __name__ == "__main__":
    main()
    