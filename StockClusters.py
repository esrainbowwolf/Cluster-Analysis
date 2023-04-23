import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import  pylab as pl
import numpy as np
#Loading the data
reddit_posts = pd.read_csv('C:/Users/video/Desktop/School Stuff/Machine Learning/Project Data/finalData.csv')

features=["sentiment","open","close","net"]
#features=["sentiment","net"] #When you want a just sentiment and net worth to be the scaling factor
reddit_posts = reddit_posts.dropna(subset=features)
data=reddit_posts[features].copy()
#Scale data with linear scaling
data = (data - data.min())/(data.max()-data.min())*10+1
#data['sentiment']=data['sentiment'].apply(lambda x: x*2) #used for making sentiment column have double the weight
#data['net']=data['net'].apply(lambda x: x*2) #When you want to scale double the weight based on net amount
#colors of the table and you can add/change whatever you want
colors = ["#00FFFF","#FF00BF","#2EFE2E","#FAAC58","#DF013A"]
#number of centroids
k=5
#Initialize random centroids
def random_centroids(data,k):
    centroids = []
    #k=number of centroids
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

centroids=random_centroids(data,3)
#Label each data point
def get_label(data,centroids):
    distances=centroids.apply(lambda x: np.sqrt(((data-x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)
labels=get_label(data,centroids) #labels each stock's cluster number
    #labels.value_counts() counts how many times each unique value occurs in a column
#Update centroids
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

#Repeat previous 2 steps until centroids stop changing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.lines import Line2D  
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), markerfacecolor=color, markersize=10) for i, color in enumerate(colors)]
def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=[colors[label] for label in labels])
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1], color='#000000')
    plt.legend(handles=legend_elements, loc='lower right')
    plt.show()

max_iterations = 100

centroids=random_centroids(data,k)
old_centroids = pd.DataFrame()
iteration=1


while iteration<max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_label(data, centroids)
    centroids = new_centroids(data, labels, k)
    plot_clusters(data, labels, centroids, iteration)
    iteration +=1
print(centroids) #uncomment to see the data shown by the charts
#The following print will show the top stock posts based on our features
#labels==0 means it is the first clust group but change the 0 to see the other cluster groups
print(reddit_posts[labels==0][["id"]+["BorS"]+features])
#---------Creating clusters using Kmeans. The stuff above is more "from scratch" clusters
#The above stuff allows us to define more in detail what we want
#The stuff below is more "optimized" according to those that created KMeans
''' kmeans = KMeans(3)
kmeans.fit(data)
KMeans(n_clusters=3)
centroids = kmeans.cluster_centers_
pd.DataFrame(centroids, columns=features).T
print(centroids) '''