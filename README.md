# Cluster-Analysis
Machine learning project for clustering based on the kmeans that already exists in python
An example of the kmeans in python is at the bottom of the code commented out.
The point of typing out this code instead of using kmeans is that this allows for more customizability.
I have also included the dataset that was used for this example that was pulled from multiple different datasets from kaggle for you to test and learn with.

This program is for clustering data sets from a csv file.
To get your file into the program look at line 7 and put in the path to your file in the same format.
To set the columns from the program you can look at line 9.
The features is the column names chosen to use for the clustering.

Make sure to follow the comments to guide you through the code.

Important lines of code:
14: uses linear scaling to scale the data to have equal weighting with one another
18: chooses the colors used for the clustering
20: decides on how many centroids to use (must have =< # of centroids to # of colors)
58: max_iterations chooses the max number of iterations possible (should never need more than 100 so you'll probably never change this)
65: loops until the new centroid created looks like the old centroid or while which iteration you are currently on is not more than max_iterations calling all your functions (main running of the code)
71: prints the example data of all the centroids
74: prints the centroid that is chosen by labels==0 (change 0 to which centroid you print with them being 0 to [# of centroids minus 1])

Functions:
1. random_centroids
    sets the centers of the centroids at random positions on the graph
2. get_label
    labels each of the data points for use
3. new_centroids
    creates a new updated centroid
4. plot_clusters
    plots clusters using the import pyplot
    pca = PCA(n_components=2) sets the size you want your dataset to be on the chart (we do 2 because we want a linear chart with x,y)
    data_2d = pca.fit_transform(data) takes the n_components size and resizes your data to fit this size (it does this using an eigen vector to get eigen values making the x,y labels irrelevant)
    clear_output(wait=True) clears the popup that appears (currently does not do anything in Visual studio Code and I do not fully understand why. You must close the popup yourself.)
    title sets the title
    first scatter plots the data points on the chart
    second scatter plots the centroids center point
    legend creates the legend with the color data
    show shows the chart
