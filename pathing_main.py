import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from numpy.random import randint
from numpy.random import seed
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import distance_matrix
import random as rn
from numpy.random import choice as np_choice
import seaborn as sns
import os

###################################
#THIS FINDS THE CENTERS OF CLUSTERS 
###################################

def center_clusters(features,x,y,z):
    
    db = DBSCAN(eps=20,min_samples=4).fit(features)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_names = list(set(labels))
    cluster_names = cluster_names[0:-1]
    label_col = labels.reshape(features.shape[0],1)
    features_labeled = np.concatenate((x,y,z,label_col), axis = 1)
    coordinates = np.zeros((n_clusters_,4))
    
    for i in range(0,len(labels)):
        
        label = round(features_labeled[i][-1])

        #if label > -1:
        for n in cluster_names:

            if n == label:

                #count of values 
                coordinates[label][3] += 1
                #x
                coordinates[label][0] += features[i][0]
                #y
                coordinates[label][1] += features[i][1] 
                #z
                coordinates[label][2] += features[i][2]
                
    for i in range(0,coordinates.shape[0]):
        
        coordinates[i][0] /= coordinates[i][3]
        coordinates[i][1] /= coordinates[i][3]
        coordinates[i][2] /= coordinates[i][3]
        
    x = coordinates[:,0]
    y = coordinates[:,1]
    z = coordinates[:,2]
    
    centers = [x,y,z]
    return x,y,z

#########################
#ANT COLONY PATH PLANNING
#########################

class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            #print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to where we started
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


#################################
#READING IN THE DATA FROM SCAN.PY
#################################

detected_locations = pd.read_csv('detected_locations.csv')

########################
#REASSIGNING THE DF NAME
########################

df_loc = detected_locations

#######################
#CONVERTING TO AN ARRAY 
#######################

df_name = np.asarray(df_loc)

#########################################################
#THIS WILL PARSE THE DF FOR INDIVIDUAL CLASSES TO CLUSTER
#########################################################


plant_centers = pd.DataFrame(data={'CLASS':[],'X':[],'Y':[],'Z':[]})

for i in range(0,7):
    
    x = []
    y = []
    z = []
    
    ################################################
    #CHECKING EACH INSTANCE IF ITS THE CURRENT CLASS
    ################################################
    
    for n in range(0,df_loc.shape[0]):
        if df_name[n][0] == i:
            x.append(df_name[n][2])
            y.append(df_name[n][3])
            z.append(df_name[n][4])
            
    ##########################################
    #IF THERE ARE INSTANCES OF A CLASS PROCEED
    ##########################################
    
    if len(x) > 0:
        
        ##########################################
        #PREPARING INPUTS FOR HEIARCHAL CLUSTERING
        ##########################################
        
        x = np.asanyarray(x).reshape(len(x),1)
        y = np.asanyarray(y).reshape(len(y),1)
        z = np.asanyarray(z).reshape(len(z),1)
        
        features = np.concatenate((x,y,z), axis = 1)
        ################################################
        #CLUSTERING THE VALUES TO FIND CENTERS OF PLANTS
        ################################################
        
        c_x,c_y,c_z = center_clusters(features,x,y,z)

        ############################################################
        #IF THERE IS DETECTED PLANTS FOUND APPENDING THEIR LOCATIONS
        ############################################################
        
        if c_x.shape[0] > 0:
            
            #CREATING CLASSES TO COINCIDE WITH THE COORDINATES
            
            CLASSES = []
            for a in range(0,c_x.shape[0]):
                CLASSES.append(i)
            CLASSES = np.asarray(CLASSES)
            tmp_df = pd.DataFrame({'CLASS':CLASSES,'X':c_x,'Y':c_y,'Z':c_z})
            plant_centers = plant_centers.append(tmp_df,ignore_index=True)
            

###########################################
#EXTRACTING THE X,Y,Z CENTERS OF THE PLANTS
###########################################

x = plant_centers.iloc[:,1].values
y = plant_centers.iloc[:,2].values
z = plant_centers.iloc[:,3].values


#################################
#FORMATTING FOR ANT COLONY THE XY
#################################

x = np.array(x)
y = np.array(y)
matrix_points = []
for i in range(0, len(x)):
        matrix_points.append([x[i], y[i]])
        
matrix_points = np.array(matrix_points, dtype=float)  


##########################################################
#COMPUTING AN EUCLIDEAN DISTANCE MATRIX BETWEEN ALL POINTS
##########################################################

dist_points = scipy.spatial.distance_matrix(matrix_points, matrix_points)
#CHANGING 0'S IN THE ARRAY TO INF VALUE
for i in range(0,dist_points.shape[1]):
    for n in range(0,dist_points.shape[0]):
        if dist_points[i,n] == 0:
            dist_points[i,n] = np.inf
            

########################################
#RUNNING THE ANT COLONY PATHING FUNCTION 
########################################
ant_colony = AntColony(dist_points, 1, 1, 100, 0.95, alpha=1, beta=1)
shortest_path = ant_colony.run()
best_path = np.asarray(shortest_path[0:len(shortest_path)-1])


#################################################
#EXTRACTING WHICH XY POINTS TO TRAVEL TO IN ORDER
#################################################

path_x = []
path_y = []
for i in range(0,matrix_points.shape[0]):
       first_index = best_path[0,i][0]
       path_x.append(matrix_points[first_index,0])
       path_y.append(matrix_points[first_index,1]) 
        
path_x = np.asarray(path_x)
path_y = np.asarray(path_y) 


###########################################
#REASSIGNING THE CLASSES TO THE COORDINATES
###########################################

center_classes = plant_centers.iloc[:,0].values
center_classes = np.asarray(center_classes)
ordered_classes = []

for i in range(0,len(x)):
    for n in range(0,len(x)):
        if path_x[i] == x[n]:
            if path_y[i]==y[n]:
                ordered_classes.append(round(center_classes[n]))

###########################################################
#WRITING THE ORDERED LOCATIONS OF PLANTS TO TRAVEL TO A CSV
###########################################################

if os.path.exists('Ordered_Plant_Locations.csv'):
    os.remove('Ordered_Plant_Locations.csv')

pathed_data = {'X':path_x,
                'Y':path_y,
                'Class':ordered_classes}
pathed_df = pd.DataFrame(pathed_data)
pathed_df.to_csv('Ordered_Plant_Locations.csv')

print('CENTERS AND PATHS BETWEEN PLANTS ARE FOUND AND IN: Ordered_Plant_Locations.csv !!')