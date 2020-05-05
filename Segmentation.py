import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as pe
from plotly.offline import plot
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from kmodes.kmodes import KModes
import logging
# import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/df.csv')
std_cons = pd.read_csv('data/Consumption_scaler.csv')
std_value = pd.read_csv('data/Value_scaler.csv')
Consumption = pd.read_csv('data/Consumption_clean.csv')
Value = pd.read_csv('data/Value_clean.csv')
outliers = pd.read_csv('data/Outliers.csv')


# 12. Consumption - K-means followed by Hierarchical clustering
# 12.1 K-means
# Define k seeds
k = 100
# Apply algorithm with k clusters
kmeans_cons = KMeans(n_clusters=k, init='k-means++', n_init=65, max_iter=300).fit(std_cons)
# 13.1.1 Cluster Evaluation
# Get centroids and save them as a dataframe
centroids_cons = kmeans_cons.cluster_centers_
inter_clus_dist = euclidean_distances(centroids_cons)
avg_inter_clus_dist = (sum(sum(inter_clus_dist)) / 2) / k
# print('Average distance intra_cluster:', k_means.inertia_ / len(std_cons))
print('Average distance inter_cluster:', avg_inter_clus_dist)

centroids_cons = pd.DataFrame(centroids_cons, columns=Consumption.columns)
clusters_cons = pd.DataFrame(kmeans_cons.labels_, columns=['Centroids'])
clusters_cons['ID'] = Consumption.index

# =============================================================================
# # Save centroids
# centroids_cons.to_csv("data/Centroids.csv")
# =============================================================================

# 12.2 Hierarchical clustering on top of K-means
# Plot the dendrogram to decide number of clusters

# Z = linkage(centroids_cons, method='ward')
# dendrogram(Z)
# Aplly hierarchical clustering with number of clusters defined and best method (ward)
Hclustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
my_HC = Hclustering.fit(centroids_cons)
my_labels = pd.DataFrame(my_HC.labels_).reset_index()
my_labels.columns = ['Centroids', 'Cluster']
# Check number of k-means centroids by hierachical clusters
count_centroids = my_labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

# Create 'final_df' with column 'Cluster' identifying to each cluster the observations belong to
final_df = clusters_cons.merge(my_labels, how='inner', on='Centroids')
final_df = std_cons.merge(final_df, how='inner', left_on=std_cons.index, right_on='ID')
final_df = final_df.set_index('ID')
final_df = final_df.drop(columns='Centroids')
# Centroids from the hierarchical clustering
centroids_HC = final_df.groupby(['Cluster']).mean()
# Check number of observations by cluster
count_HC = final_df.Cluster.value_counts()
count_HC = final_df.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

# 13. Value Analysis
# 13.1. K-Means
# Define k seeds
k = 50
# Aplly kmeans with number of clusters defined
k_means = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=300).fit(std_value)
clusters_value = pd.DataFrame(k_means.labels_, columns=['Cluster'])
clusters_value['ID'] = Value.index

# 13.1.1 Cluster Evaluation
# Get centroids and save them as a dataframe
clus_center = k_means.cluster_centers_
inter_clus_dist = euclidean_distances(clus_center)
avg_inter_clus_dist = (sum(sum(inter_clus_dist)) / 2) / k
print('Average distance intra_cluster:', k_means.inertia_ / len(std_value))
print('Average distance inter_cluster:', avg_inter_clus_dist)

# 13.2 Hierarchical clustering on top of K-means
# Plot the dendrogram to decide number of clusters

logging.getLogger('matplotlib.font_manager').disabled = True
Z = linkage(clus_center, method='ward')
dendrogram(Z, p=50)
# Aplly hierarchical clustering with number of clusters defined and best method (ward)
Hclustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
my_HC = Hclustering.fit(std_value)
my_labels = pd.DataFrame(my_HC.labels_).set_index(Value.index)
my_labels.columns = ['Labels']

# Create 'final_df_hc' with column 'Cluster' identifying to each cluster the observations belong to
final_df_hc = pd.concat([pd.DataFrame(std_value), my_labels], axis=1)
final_df_hc.columns = ['ID','Years_Education', 'Salary_Invested', 'CMV', 'Cluster']

# Check number of observations by cluster
final_df_hc.groupby(['Cluster']).count()

# Centroids from the hierarchical clustering with non-standardized values
scaler = StandardScaler()
to_revert = final_df_hc.groupby(['Cluster'])['Years_Education', 'Salary_Invested', 'CMV'].mean()
# final_result = pd.DataFrame(scaler.inverse_transform(X=to_revert),
#                             columns=['Years_Education', 'Salary_Invested', 'CMV'])

# 14. SOM
# 14.1. SOM FOR CONSUMPTION
# Define SOM grid size
mapsize_consump = 9

# Create algorithm, define parameters and train the grid
sm_consump = SOMFactory().build(data=std_cons.values,
                                mapsize=(mapsize_consump, mapsize_consump),
                                normalization='var',
                                initialization='random',
                                component_names=Consumption.columns,
                                lattice='rect',
                                training='batch')
sm_consump.train(n_job=6,
                 verbose='info',
                 train_rough_len=35,
                 train_finetune_len=80)

# 'final_clusters_consump' is a dataframe similar to df but including a column 'Labels' which indicates the closest neuron to each obs
final_clusters_consump = pd.DataFrame(sm_consump._data, columns=Consumption.columns).set_index(Consumption.index)
my_labels_c = pd.DataFrame(sm_consump._bmu[0], columns=['Labels']).set_index(Consumption.index)
final_clusters_consump = pd.concat([final_clusters_consump, my_labels_c], axis=1)

# Plot the number of observations associated to each neuron
vhts_c = BmuHitsView(12, 12, "Hits Map", text_size=7)
vhts_c.show(sm_consump, anotate=True, onlyzeros=False, labelsize=10, cmap="summer", logaritmic=False)
plt.show()

# Visualization of the value of the grid neurons in each variable
view2D_c = View2D(9, 9, "", text_size=7)
view2D_c.show(sm_consump, col_sz=5, what='codebook')

# 14.1.1. Hierarchical Clustering on top of SOM
# Create a dataframe that specifies the coordinates of the neurons
grid_consump = final_clusters_consump.groupby(by='Labels').mean()
labels_som_cons = grid_consump.index

# Plot a dendrogram to choose the number of clusters
Z = linkage(grid_consump, method='ward')  # method='single', 'complete', 'ward'
dendrogram(Z, p=30)

# Aplly hierarchical clustering with k clusters defined and best method (ward)
k = 3
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
my_HC_consump = Hclustering.fit(grid_consump)

# 'som_hc_cons' is a dataframe with the column 'Hierarchical Clustering' that specifies to which cluster belongs each client
som_hc_cons = pd.DataFrame(my_HC_consump.labels_, columns=['Hierarchical Clustering'])
som_hc_cons['Labels'] = labels_som_cons
som_hc_cons = final_clusters_consump.merge(som_hc_cons, how='inner', on='Labels', right_index=True)
som_hc_cons = som_hc_cons.sort_index()

# Verify the number of observations associated of each cluster and the cluster centroids coordinates
count_obs_som_hc_cons = som_hc_cons.groupby('Hierarchical Clustering').count()
centroids_hc_som_cons = som_hc_cons.groupby('Hierarchical Clustering').mean()
centroids_hc_som_cons = centroids_hc_som_cons.drop(columns='Labels')

# 14.1.1.1 Silhouette scores
# Average silhouette score
silhouette_avg_som_hc_cons = silhouette_score(std_cons, som_hc_cons['Hierarchical Clustering'].values)
# Silhouette scores individual to each observation
sample_silhouette_som_hc_cons = pd.DataFrame(
    silhouette_samples(std_cons, som_hc_cons['Hierarchical Clustering'].values), columns=['Value'])
# Number of positives silhouette scores
pos_sample_hc_cons = sample_silhouette_som_hc_cons[sample_silhouette_som_hc_cons.Value > 0].count()

# 14.1.2 K-Means Clustering on top of SOM
# Visualize to which of the k cluster from the k-means belongs each neuron
k = 3
som_kmeans_cons = sm_consump.cluster(k)
hits = HitMapView(10, 10, "Clustering", text_size=7)
a = hits.show(sm_consump)

# 'som_kmeans_cons' is a dataframe with a column 'K-means' that specifies to which cluster belongs each client
som_kmeans_cons = pd.DataFrame(som_kmeans_cons, columns=['K_means'])
som_kmeans_cons['Labels'] = range(mapsize_consump * mapsize_consump)
som_kmeans_cons = final_clusters_consump.merge(som_kmeans_cons, how='inner', on='Labels', right_index=True)
som_kmeans_cons = som_kmeans_cons.sort_index()

# Verify the number of observations associated of each cluster and the cluster centroids coordinates
count_obs_som_kmeans_cons = som_kmeans_cons.groupby('K_means').count()
centroids_som_kmeans_cons = som_kmeans_cons.groupby('K_means').mean()
centroids_som_kmeans_cons = centroids_som_kmeans_cons.drop(columns='Labels')

# 14.1.2.1 silhouette scores
# Average silhouette score
silhouette_avg_som_k_means_cons = silhouette_score(std_cons, som_kmeans_cons['K_means'].values)
# Silhouette scores individual to each observation
sample_silhouette_som_k_means_cons = pd.DataFrame(silhouette_samples(std_cons, som_kmeans_cons['K_means'].values),
                                                  columns=['Value'])
# Number of positives silhouette scores
pos_sample_kmeans_cons = sample_silhouette_som_k_means_cons[sample_silhouette_som_k_means_cons.Value > 0].count()

# 14.2 SOM FOR VALUE
# Define SOM grid size
mapsize = 7

# Create algorithm, define parameters and train the grid
Value_non_discrete = Value[['Years_Education', 'Salary_Invested', 'CMV']]
std_value = std_value.set_index('ID')
sm_value = SOMFactory().build(data=std_value.values, mapsize=(mapsize, mapsize), normalization='var',
                              initialization='random',
                              component_names=Value_non_discrete.columns, lattice='rect', training='seq')
sm_value.train(n_job=6, verbose='info', train_rough_len=30, train_finetune_len=70)

# 'final_clusters_value' is a dataframe similar to df but including a column 'Labels' which indicates the closest neuron to each obs
final_clusters_value = pd.DataFrame(sm_value._data, columns=Value_non_discrete.columns).set_index(Value_non_discrete.index)
my_labels_v = pd.DataFrame(sm_value._bmu[0], columns=['Labels']).set_index(final_clusters_value.index)
final_clusters_value = pd.concat([final_clusters_value, my_labels_v], axis=1)

# Plot the number of observations associated to each neuron
vhts_v = BmuHitsView(10, 10, "Hits Map", text_size=7)
vhts_v.show(sm_value, anotate=True, onlyzeros=False, labelsize=12, cmap="spring", logaritmic=False)

# Visualization of the value of each neuron in each variable
view2D_v = View2D(7, 7, "", text_size=7)
view2D_v.show(sm_value, col_sz=4, what='codebook', which_dim="all")
plt.show()

# 14.2.1. HIERARCHICAL CLUSTERING ON TOP OF SOM
# Create a dataframe that specifies the coordinates of the neurons
grid_v = final_clusters_value.groupby(by='Labels').mean()
labels_som = grid_v.index

# Plot a dendrogram to choose the number of clusters
Z = linkage(grid_v, method='ward')  # method='single', 'complete', 'ward'
dendrogram(Z, p=49)

# Aplly hierarchical clustering with k clusters defined and best method (ward)
k = 4
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
my_HC = Hclustering.fit(grid_v)

# 'som_hc_value' is a dataframe with the column 'Hierarchical Clustering' that specifies to which cluster belongs each client
som_hc_value = pd.DataFrame(my_HC.labels_, columns=['Hierarchical Clustering'])
som_hc_value['Labels'] = labels_som
som_hc_value = final_clusters_value.merge(som_hc_value, how='inner', on='Labels', right_index=True)
som_hc_value = som_hc_value.sort_index()

# Verify the number of observations associated of each cluster and the cluster centroids coordinates
count_obs_som_hc_value = som_hc_value.groupby('Hierarchical Clustering').count()
centroids_hc_som_value = som_hc_value.groupby('Hierarchical Clustering').mean()
centroids_hc_som_value = centroids_hc_som_value.drop(columns='Labels')

# 14.2.1.1 Silhouette Analysis
# Average silhouette score
silhouette_avg_som_hc_value = silhouette_score(std_value, som_hc_value['Hierarchical Clustering'].values)
# Silhouette scores individual to each observation
sample_silhouette_som_hc_value = pd.DataFrame(
    silhouette_samples(std_value, som_hc_value['Hierarchical Clustering'].values), columns=['Value'])
# Number of positives silhouette scores
pos_sample_hc_value = sample_silhouette_som_hc_value[sample_silhouette_som_hc_value.Value > 0].count()

# 14.2.2 k-MEANS CLUSTERING ON TOP OF SOM
# Visualize to which of the k cluster from the k-means belongs each neuron
k = 3
som_kmeans_value = sm_value.cluster(k)
hits = HitMapView(7, 7, "Clustering", text_size=7)
a = hits.show(sm_value)

# 'som_kmeans_value' is a dataframe with a column 'K-means' that specifies to which cluster belongs each client
som_kmeans_value = pd.DataFrame(som_kmeans_value, columns=['K_means'])
som_kmeans_value['Labels'] = range(mapsize * mapsize)
som_kmeans_value = final_clusters_value.merge(som_kmeans_value, how='inner', on='Labels', right_index=True)
som_kmeans_value = som_kmeans_value.sort_index()

# Verify the number of observations associated of each cluster and the cluster centroids coordinates
count_obs_som_kmeans_value = som_kmeans_value.groupby('K_means').count()
centroids_som_kmeans_value = som_kmeans_value.groupby('K_means').mean()
centroids_som_kmeans_value = centroids_som_kmeans_value.drop(columns='Labels')

# 14.2.2.1 Silhouette Analysis
# Average silhouette score
silhouette_avg_som_k_means_value = silhouette_score(std_value, som_kmeans_value['K_means'].values)
# Silhouette scores individual to each observation
sample_silhouette_som_k_means_value = pd.DataFrame(silhouette_samples(std_value, som_kmeans_value['K_means'].values),
                                                   columns=['Value'])
# Number of positives silhouette scores
pos_sample_kmeans_value = sample_silhouette_som_k_means_value[sample_silhouette_som_k_means_value.Value > 0].count()

# =============================================================================
# # 14.3 SAVE BEST DATA SETS OF SOM
# # Value
som_hc_value.to_csv('data/value_obs_HC.csv')
# centroids_hc_som_value.to_csv('data/centroids_HC.csv')
# som_kmeans_value.to_csv('data/obs_kmeans.csv')
# centroids_som_kmeans_value.to_csv('data/centroids_kmeans.csv')
# # Consumption
som_hc_cons.to_csv('data/consumption_obs_hc.csv')
# centroids_hc_som_cons.to_csv('data/centroids_hc.csv')
# som_kmeans_cons.to_csv('data/obs_kmeans.csv')
# centroids_som_kmeans_cons.to_csv('data/centroids_kmeans.csv')
#
# =============================================================================

# 14.4 SOM VISUALIZATION OF THE BEST SOLUTIONS
# FOR SOM VALUE VISUALIZATION WITH HC
v = 'Hierarchical Clustering'

# Import obsevation with cluster labels (best soutions choosed)
clusters_v = pd.read_csv('data/value_obs_HC.csv')

# 3d plot with clusters in 'Value' subset
fig = plt.figure()
ax = Axes3D(fig)
x_points = clusters_v['Years_Education']
z_points = clusters_v['Salary_Invested']
y_points = clusters_v['CMV']
ax.scatter3D(x_points, y_points, z_points, c=clusters_v[v], cmap='plasma');
ax.set_xlabel("Years_Education")
ax.set_ylabel("Salary_Invested")
ax.set_zlabel('CMV')
plt.show()

# FOR SOM CONSUMPTION VISUALIZATION WITH HC

# Import obsevation with cluster labels (best soutions choosed)
c = 'Hierarchical Clustering'
clusters_c = pd.read_csv('data/consumption_obs_hc.csv')

# 3d plot with clusters in 'Consumption' subset
# Once there are five variables used in 'Consumption' subset and we can only plot three, one can choose the ones that he wants to visualize
fig = plt.figure()
ax = Axes3D(fig)
x = 'Motor_Share'
y = 'Health_Share'
z = 'Work_Share'
# x = 'Life_Share'
# x = 'Household_Share'
x_points = clusters_c[x]
z_points = clusters_c[z]
# z_points = clusters_c['Life_Share']
# z_points = clusters_c['Household_Share']
y_points = clusters_c[y]

ax.scatter3D(x_points, y_points, z_points, c=clusters_c[c], cmap='plasma');
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
plt.show()

# 15. KMODES
# 15.1. Elbow Graph
# cost will append the error/inertia of the application of K-modes to the 'value modes' dataset with 1 to max_k number of clusters
cost = []
max_k = 10
# 'Value Modes' is a dataset with the non-numerical variables to use in K-modes
value_modes = Value[['Children', 'Area', 'Abandoned']].astype('str')
for i in range(1, max_k + 1):
    k = i
    km = KModes(n_clusters=k, init='random', n_init=25, verbose=1)
    clusters_kmodes = km.fit_predict(value_modes)
    clusters_kmodes = pd.DataFrame(clusters_kmodes)
    clusters_kmodes.columns = ['Cluster_Kmodes']
    clusters_kmodes['ID'] = value_modes.index
    clusters_kmodes.set_index('ID', inplace=True)
    cost.append(km.cost_)

# Plot the elbow graph to decide between the number of clusters
fig = plt.figure()
ax = plt.axes(xlim=(1, k),
              ylim=(min(cost),
                    max(cost)))
comp = []
for i in range(1, k + 1):
    comp.append(i)

plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Graphic')
plt.plot(comp, cost, 'bx-')
plt.xticks(range(1, k + 1))
plt.show()

# 15.2 Kmodes
# Apply algorithm with k clusters
k = 5
km = KModes(n_clusters=k, init='random', n_init=75, verbose=1)
clusters_kmodes = km.fit_predict(value_modes)

# 'clusters_kmodes' is a dataframe where 'Cluster_Kmodes' column indicates the group that each observation belongs to
clusters_kmodes = pd.DataFrame(clusters_kmodes)
clusters_kmodes.columns = ['Cluster_Kmodes']
clusters_kmodes['ID'] = value_modes.index
clusters_kmodes.set_index('ID', inplace=True)

# 15.3 Check KModes clusters

df_kmodes_aux = clusters_kmodes.merge(value_modes, how='inner', on= clusters_kmodes.index)
a = df_kmodes_aux.groupby(['Cluster_Kmodes']).count()
df_kmodes_aux[df_kmodes_aux['Area'].eq('1.0') & df_kmodes_aux['Cluster_Kmodes'].eq(0)]
df_kmodes_aux[df_kmodes_aux['Children'].eq('1') & df_kmodes_aux['Cluster_Kmodes'].eq(0)]
df_kmodes_aux[df_kmodes_aux['Abandoned'].eq('1') & df_kmodes_aux['Cluster_Kmodes'].eq(0)]

df_kmodes_area = pd.crosstab(df_kmodes_aux['Cluster_Kmodes'], df_kmodes_aux['Area']).apply(lambda x: x / x.sum(),
                                                                                           axis=1)
df_kmodes_child = pd.crosstab(df_kmodes_aux['Cluster_Kmodes'], df_kmodes_aux['Children']).apply(lambda x: x / x.sum(),
                                                                                                axis=1)
df_kmodes_abandoned = pd.crosstab(df_kmodes_aux['Cluster_Kmodes'], df_kmodes_aux['Abandoned']).apply(
    lambda x: x / x.sum(), axis=1)
df_kmodes = pd.concat([df_kmodes_area, df_kmodes_child], axis=1)
df_kmodes = pd.concat([df_kmodes, df_kmodes_abandoned], axis=1)
df_kmodes.columns = ['Area 1', 'Area 2', 'Area 3', 'Area 4', '~ Children (=0)', 'Children (=1)', '~ Abandoned (=0)',
                     'Abandoned (=1)']
df_kmodes['Size'] = a['Children']
df_kmodes.groupby('Cluster_Kmodes').count()

# Merge Final Datasets
df_kmodes_aux.columns = ['ID','Cluster_Kmodes','Children','Area','Abandoned']
final_df = final_df_hc.merge(df_kmodes_aux, how='inner', on='ID')
final_df.groupby(['Cluster'])['Cluster_Kmodes'].count()
final_df_cross = pd.crosstab(final_df['Cluster'], final_df['Cluster_Kmodes']).apply(lambda x: x / x.sum(), axis=1)

# 16. DBSCAN
# std_cons.columns = ['Motor_Share', 'Household_Share', 'Health_Share', 'Life_Share', 'Work_Share']
# db = DBSCAN(eps=1, min_samples=10).fit(std_cons)
#
# labels = db.labels_
# # Number of clusters in labels, ignoring noise if present
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# unique_clusters, counts_clusters = np.unique(db.labels_, return_counts=True)
# c = np.asarray((unique_clusters, counts_clusters))

# 17. Merge clusters
# Import datasets with final clusters by subset
df_value = pd.read_csv('data/value_obs_HC.csv')
df_cons = pd.read_csv('data/consumption_obs_HC.csv')
df_value.rename(columns={'Hierarchical Clustering': 'Cluster_value'}, inplace=True)
df_cons.rename(columns={'Hierarchical Clustering': 'Cluster_cons'}, inplace=True)
df_value.drop(columns='Labels', inplace=True)
df_cons.drop(columns='Labels', inplace=True)

# Concatenate both subsets 'Value' and 'Consumption' in one: 'cluster_df'
cluster_df = pd.concat([df_value, df_cons], axis=1)

# Create contigency table with number of observation by each of the 12 clusters generated by the 3*4 clusters from each subset 'Value' and 'Consumption'
crosstab = pd.crosstab(cluster_df.Cluster_value, cluster_df.Cluster_cons)

cluster_df.loc[(cluster_df['Cluster_value'] == 0) & (cluster_df['Cluster_cons'] == 0), 'Cluster'] = 0
cluster_df.loc[(cluster_df['Cluster_value'] == 1) & (cluster_df['Cluster_cons'] == 0), 'Cluster'] = 1
cluster_df.loc[(cluster_df['Cluster_value'] == 2) & (cluster_df['Cluster_cons'] == 0), 'Cluster'] = 2
cluster_df.loc[(cluster_df['Cluster_value'] == 0) & (cluster_df['Cluster_cons'] == 1), 'Cluster'] = 3
cluster_df.loc[(cluster_df['Cluster_value'] == 1) & (cluster_df['Cluster_cons'] == 1), 'Cluster'] = 4
cluster_df.loc[(cluster_df['Cluster_value'] == 2) & (cluster_df['Cluster_cons'] == 1), 'Cluster'] = 5
cluster_df.loc[(cluster_df['Cluster_value'] == 0) & (cluster_df['Cluster_cons'] == 2), 'Cluster'] = 6
cluster_df.loc[(cluster_df['Cluster_value'] == 1) & (cluster_df['Cluster_cons'] == 2), 'Cluster'] = 7
cluster_df.loc[(cluster_df['Cluster_value'] == 2) & (cluster_df['Cluster_cons'] == 2), 'Cluster'] = 8
cluster_df.loc[(cluster_df['Cluster_value'] == 0) & (cluster_df['Cluster_cons'] == 3), 'Cluster'] = 9
cluster_df.loc[(cluster_df['Cluster_value'] == 1) & (cluster_df['Cluster_cons'] == 3), 'Cluster'] = 10
cluster_df.loc[(cluster_df['Cluster_value'] == 2) & (cluster_df['Cluster_cons'] == 3), 'Cluster'] = 11

# 17.1 Merge by similarity

end_df = cluster_df.drop(columns=['Cluster_value', 'Cluster_cons']).copy()
centroids_final = end_df.groupby('Cluster').mean()

distances = pdist(centroids_final.values, metric='euclidean')
dist_matrix = pd.DataFrame(squareform(distances), columns=centroids_final.index.values.astype(str),
                           index=centroids_final.index.values)
count_obs = end_df.groupby('Cluster').count()['Years_Education']

i = 0
while count_obs.min() < 630:
    # get smallest cluster
    small_clus = count_obs[count_obs == count_obs.min()].index[0]
    # get nearest cluster to smallest
    nearest_clus = \
    dist_matrix.loc[dist_matrix[str(small_clus)] == sorted(dist_matrix.loc[:, str(small_clus)])[1]].index[0]
    # Join 'small_clus' obs to 'nearest_clus' obs
    end_df.loc[end_df['Cluster'] == small_clus, 'Cluster'] = nearest_clus
    # Recalculate centroids
    centroids_final = end_df.groupby('Cluster').mean()
    # Recalculate distance matrix
    distances = pdist(centroids_final.values, metric='euclidean')
    dist_matrix = pd.DataFrame(squareform(distances), columns=centroids_final.index.values.astype(str),
                               index=centroids_final.index.values)
    # Recalculate number of observations associated to each cluster
    count_obs = end_df.groupby('Cluster').count()['Years_Education']

    i = i + 1

# 17.2. Gaussian Mixture
# end_df = cluster_df.drop(columns=['Cluster'])
# gmm = mixture.GaussianMixture(n_components= 12, #<the number of elements you found before> <number of centroids>,
#                              init_params='kmeans', # {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
#                              max_iter=1000,
#                              n_init=10,
#                              verbose = 1)
#
# gmm.fit(end_df)
# EM_labels_ = pd.DataFrame(gmm.predict(end_df))
# EM_labels_.columns = ['Labels']
# EM_labels_['ID'] = cluster_df.index
# EM_labels_ = EM_labels_.set_index('ID')
# EM_labels_ = pd.concat([EM_labels_,cluster_df.Cluster], axis=1)
#
# crosstab2 = pd.crosstab(EM_labels_.Labels, EM_labels_.Cluster)
#
#
# def join_clusters (df1, df2, df3, column):
#    for i in df3.loc[:,column]:
#        if i != 0:
#           row = int(df3[df3.loc[:,column]==i].index[0]) # get row(index) of the i !=0
#          # row_max = max(df.loc[row,:])
#           cluster = list(df3.loc[row,:]).index(max(df3.loc[row,:])) # get the column of the max value in 'row'
#
#           temp_df = df2.loc[(df2['Cluster']==column) & (df2['Labels']==row)].reset_index()
#
#           df1.loc[df1.index.isin(temp_df['ID']), 'Cluster'] = cluster
#
# join_clusters(cluster_df,EM_labels_,crosstab2,9)

# 18. DATA VISUALIZATION
end_df.drop(columns =['Unnamed: 0'],inplace = True)
graph1 = end_df.groupby('Cluster')[
    'Motor_Share', 'Household_Share', 'Health_Share', 'Life_Share', 'Work_Share'].mean().reset_index()
graph1 = graph1.melt(id_vars='Cluster', var_name='Variable', value_name='Average')
graph1['Cluster'] = graph1['Cluster'].astype('int64')
fig = pe.line_polar(graph1, r="Average", theta="Variable", color="Cluster", line_close=True,
                    color_discrete_sequence=["blue", "green", "red", "yellow", "orange"])
plot(fig)

# df.set_index('ID', inplace=True)
end_df = pd.concat([end_df, df[['Education', 'Area', 'Children', 'Abandoned']]], axis=1)  # re-add categorical variables

end_df.loc[(end_df['Cluster'] == 5), 'Cluster'] = 4

end_df.loc[(end_df['Cluster'] == 4), 'Segment'] = 'Silver Motor'
end_df.loc[(end_df['Cluster'] == 8), 'Segment'] = 'Gold Household vs Motor'
end_df.loc[(end_df['Cluster'] == 2), 'Segment'] = 'Bronze Work & Life'
end_df.loc[(end_df['Cluster'] == 1), 'Segment'] = 'Bronze Health'

bx = sb.boxplot(x="Segment", y="CMV", data=end_df)

N_cluster = end_df.Segment.value_counts().reset_index()
N_cluster.columns = ['Segment', 'N']

fig = pe.pie(N_cluster, values='N', names='Segment')
plot(fig)

# 19. RE-ADD OUTLIERS THROUGH CLASSIFICATION TREES
# 19.1. Select dependent and target variables
# Select variables used previous in clustering with values non-standardized as dependent variables
X = end_df[['Years_Education', 'Motor_Share', 'Household_Share', 'Health_Share']]
# Select target variable
Y = end_df[['Cluster']]

# 19.2. Transform Variables For Outliers
# outliers = outliers.set_index('ID')
outliers['Education'] = outliers['Education'].str[4:]
# Transform the ordinal variable education in a numeric one
outliers.loc[outliers['Education'] == 'Primary', 'Years_Education'] = 5
outliers.loc[outliers['Education'] == 'Basic', 'Years_Education'] = 9
outliers.loc[outliers['Education'] == 'High School', 'Years_Education'] = 12
outliers.loc[outliers['Education'] == 'BSc/MSc', 'Years_Education'] = 16
outliers.loc[outliers['Education'] == 'PhD', 'Years_Education'] = 20
outliers['Total_Premiums'] = outliers['Motor'] + outliers['Household'] + outliers['Health'] + outliers['Life'] + \
                             outliers['Work_Compensations']
outliers['Salary_Invested'] = outliers['Total_Premiums'] / (outliers['Salary'] * 12) * 100
# Create relative premiums
outliers['Motor_Share'] = outliers['Motor'] / outliers['Total_Premiums'] * 100
outliers['Household_Share'] = outliers['Household'] / outliers['Total_Premiums'] * 100
outliers['Health_Share'] = outliers['Health'] / outliers['Total_Premiums'] * 100
outliers['Life_Share'] = outliers['Life'] / outliers['Total_Premiums'] * 100
outliers['Work_Share'] = outliers['Work_Compensations'] / outliers['Total_Premiums'] * 100

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
# clf = clf.fit(X_train, y_train)
#
# # Check variables importance
# clf.feature_importances_
# pred = outliers[['Years_Education', 'Motor_Share', 'Household_Share', 'Health_Share']]
# clf.predict(pred)
# plot_tree(clf, filled=True, max_depth=5, impurity=True)

