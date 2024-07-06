from scipy.spatial import distance

def distance_selection(selected_distance, list1, list2):
    if selected_distance == "Euclidean":
        dist = distance.euclidean(list1, list2)
    elif selected_distance == "Canberra":
        dist = distance.canberra(list1, list2)
    elif selected_distance == "Manhattan":
        dist = distance.cityblock(list1, list2)
    elif selected_distance == "Chebyshev":
        dist = distance.chebyshev(list1, list2)
    elif selected_distance == "Minkowsky":
        dist = distance.minkowski(list1, list2)
    return dist