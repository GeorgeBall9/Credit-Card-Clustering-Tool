import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yellowbrick.cluster import SilhouetteVisualizer
from pywaffle import Waffle
from matplotlib.lines import Line2D


class KMeansClustering:
    def __init__(self, dataset: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
        """
        K-Means Clustering class to perform clustering analysis using K-Means algorithm.

        Args:
            dataset (pd.DataFrame): The dataset to be used for clustering.
            n_clusters (int): The number of clusters to create. Selected is 5.
            random_state (int): The random seed for reproducibility. Default is 32.

        Attributes:
            dataset (pd.DataFrame): The dataset to be used for clustering.
            n_clusters (int): The number of clusters to create.
            model (KMeans): The KMeans model for clustering analysis.

        """
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=random_state)

    def inertia_plot(self, max_clusters: int = 10):
        """
        Plot the Inertia (SSE) for different numbers of clusters to determine the optimal number of clusters.

        Args:
            max_clusters (int): The maximum number of clusters to consider. Default is 10.

        """
        # Initialise an empty list to store the inertia values
        inertia = []

        # Loop over a range of cluster numbers from 1 to the maximum number of clusters
        for i in range(1, max_clusters + 1):
            # Initialise a KMeans model with the current number of clusters
            kmeans = KMeans(n_clusters=i, n_init=10)
            
            # Fit the KMeans model to the dataset
            kmeans.fit(self.dataset)
            
            # Append the inertia (SSE) of the current model to the list
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        plt.plot(range(1, max_clusters + 1), inertia, marker='o', linestyle='-', color='aquamarine')
        plt.title("Inertia Plot (SSE) / Elbow Method", fontsize=16, fontweight='bold', fontfamily='serif')
        plt.xlabel("Number of clusters", fontsize=14, fontfamily='serif')
        plt.ylabel("SSE", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("Plots/Clustering/sse_plot.png")  # save the plot as a .png file
        plt.close()  # close the plot
    
    def silhouette_analysis(self, max_clusters: int = 10):
        """
        Perform silhouette analysis to evaluate the quality of clustering for different numbers of clusters.

        Args:
            max_clusters (int): The maximum number of clusters to consider. Default is 10.

        """
        
        # Initialise an empty list to store the silhouette scores
        silhouette_scores = []
        
        # Loop over a range of cluster numbers from 2 to the maximum number of clusters
        for i in range(2, max_clusters + 1):  # start from 2 because silhouette score is not defined for 1 cluster
            # Initialize a KMeans model with the current number of clusters
            kmeans = KMeans(n_clusters=i, n_init=10)

            # Fit the KMeans model to the dataset
            kmeans.fit(self.dataset)

            # Calculate the silhouette score of the current model
            score = silhouette_score(self.dataset, kmeans.labels_)

            # Append the silhouette score to the list
            silhouette_scores.append(score)

        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='aquamarine')
        plt.title("Silhouette Score Plot", fontsize=16, fontweight='bold', fontfamily='serif')
        plt.xlabel("Number of clusters", fontsize=14, fontfamily='serif')
        plt.ylabel("Silhouette Score", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("Plots/Clustering/silhouette_score_plot.png")  # save the plot as a .png file
        plt.close()  # close the plot


    def fit_model(self):
        """
        Fit the K-Means model to the dataset.

        """
        self.model.fit(self.dataset)
        self.dataset_df = pd.DataFrame(self.dataset)
        self.dataset_df["Cluster"] = self.model.labels_

    def silhouette_score(self):
        """
        Calculate the silhouette score for the clustered data.

        Returns:
            float: The silhouette score.

        """
        score = silhouette_score(self.dataset, self.model.labels_)
        print(f"Silhouette Score: {score}")
        
    def calinski_harabasz_index(self):
        """
        Calculate the Calinski-Harabasz index for the clustered data.

        Returns:
            float: The Calinski-Harabasz index.

        """
        chi = calinski_harabasz_score(self.dataset, self.model.labels_)
        print(f"Calinski-Harabasz Index: {chi}")
        return chi

    def davies_bouldin_index(self):
        """
        Calculate the Davies-Bouldin index for the clustered data.

        Returns:
            float: The Davies-Bouldin index.

        """
        dbi = davies_bouldin_score(self.dataset, self.model.labels_)
        print(f"Davies-Bouldin Index: {dbi}")
        return dbi
    
    def cluster_properties(self):
        """
        Calculate the mean values for each cluster and display the cluster properties.

        """
        cluster_props = self.dataset_df.groupby('Cluster').mean()
        print(cluster_props)
    
        
    def visualise_clusters(self):
        """
        Visualize the clustered data using various plots and charts.

        """
        cluster_colors=sns.color_palette('mako', n_colors=self.n_clusters)  # use the 'mako' color scheme
        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids']
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        legend_style=dict(borderpad=2, frameon=False, fontsize=8)
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axs.flatten()  # flatten the 2D array of AxesSubplot objects

        # Silhouette plot
        s_viz = SilhouetteVisualizer(self.model, colors='mako', ax=ax1)
        s_viz.fit(self.dataset)
        ax1.set_xlabel("Silhouette Coefficient Values", fontfamily='serif', fontsize=8)
        ax1.set_ylabel("Data Points Grouped by Clusters", fontfamily='serif',fontsize=8)
        # Create a legend for the silhouette plot
        legend1 = Line2D([0], [0], color='red', lw=2, linestyle='--', label='Average Silhouette Score')
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color, markersize=10) for label, color in zip(labels, cluster_colors)]
        ax1.legend(handles=[legend1] + legend_elements, **legend_style)
        

        # Scatter plot of the cluster distributions
        
        # map the 'Cluster' labels in your DataFrame to the colors
        color_dict = {i: color for i, color in enumerate(cluster_colors)}
        point_colors = self.dataset_df["Cluster"].map(color_dict)
        
        ax2.scatter(
            self.dataset_df.iloc[:, 0],
            self.dataset_df.iloc[:, 1],
            s=10,
            c=point_colors,
            edgecolor=None
        )
        ax2.scatter(
            self.model.cluster_centers_[:, 0],
            self.model.cluster_centers_[:, 1],
            s=200,
            c="red",
            edgecolor='black'
        )
        # Remove x-axis and y-axis ticks as they do not provide meaningful insight
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Create a legend for the scatter plot
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) for label, color in zip(labels, cluster_colors)]
        legend2 = ax2.legend(handles=legend_elements, loc="best", **legend_style)
        ax2.add_artist(legend2)

        # Waffle chart
        ax3=plt.subplot(2, 2, (3,4))
        unique, counts = np.unique(self.model.labels_, return_counts=True)
        df_waffle = dict(zip(unique, counts))
        total = sum(df_waffle.values())
        wfl_square = {key: value/100 for key, value in df_waffle.items()}
        wfl_label = {key: round(value/total*100, 2) for key, value in df_waffle.items()}
        Waffle.make_waffle(ax=ax3, rows=6, values=wfl_square, colors=cluster_colors, 
                        labels=[f"Cluster {i+1} - ({k}%)" for i, k in wfl_label.items()], 
                        legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': 5, 'borderpad': 2, 
                                'frameon': False, 'fontsize':10, })
        ax3.text(0.01, -0.09, '** 1 square ≈ 100 customers', weight = 'bold', style='italic', fontsize=8, fontfamily='serif')
        
        ax4.axis('off')  # turn off the fourth subplot
        
        plt.suptitle('Credit Card Customer Clustering using K-Means\n', fontsize=14, **text_style)

        plt.savefig("Plots/Clustering/clusters.png")  # save the plot as a .png file
        plt.close()  # close the plot
        
    def cluster_summary(self):
        """
        Generate a summary of the clusters including overall and per-cluster statistics.

        Returns:
            pd.DataFrame: The cluster summary.

        """
        # Add the cluster labels to the dataset
        self.dataset_df["Cluster"] = self.model.labels_
        self.dataset_df["Cluster"] = 'Cluster ' + (self.dataset_df["Cluster"] + 1).astype(str)

        # Calculate overall mean
        df_profile_overall = pd.DataFrame()
        df_profile_overall['Overall'] = self.dataset_df.describe().loc[['mean']].T

        # Group by cluster labels and calculate mean
        df_cluster_summary = self.dataset_df.groupby("Cluster").describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
        df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')

        # Join both dataframes
        df_profile = df_cluster_summary.join(df_profile_overall).reset_index()

        return df_profile

   


