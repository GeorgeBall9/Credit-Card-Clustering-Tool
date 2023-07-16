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
    def __init__(self, dataset: pd.DataFrame, n_clusters: int = 5, random_state: int = 32):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=random_state)

    def inertia_plot(self, max_clusters: int = 10):
        inertia = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init=10)
            kmeans.fit(self.dataset)
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
        plt.savefig("Plots/sse_plot.png")  # save the plot as a .png file
        plt.close()  # close the plot
    
    def silhouette_analysis(self, max_clusters: int = 10):
        silhouette_scores = []
        for i in range(2, max_clusters + 1):  # start from 2 because silhouette score is not defined for 1 cluster
            kmeans = KMeans(n_clusters=i, n_init=10)
            kmeans.fit(self.dataset)
            score = silhouette_score(self.dataset, kmeans.labels_)
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
        plt.savefig("Plots/silhouette_score_plot.png")  # save the plot as a .png file
        plt.close()  # close the plot


    def fit_model(self):
        self.model.fit(self.dataset)
        self.dataset_df = pd.DataFrame(self.dataset)
        self.dataset_df["Cluster"] = self.model.labels_

    def silhouette_score(self):
        score = silhouette_score(self.dataset, self.model.labels_)
        print(f"Silhouette Score: {score}")
        
    def calinski_harabasz_index(self):
        chi = calinski_harabasz_score(self.dataset, self.model.labels_)
        print(f"Calinski-Harabasz Index: {chi}")
        return chi

    def davies_bouldin_index(self):
        dbi = davies_bouldin_score(self.dataset, self.model.labels_)
        print(f"Davies-Bouldin Index: {dbi}")
        return dbi
    
    def cluster_properties(self):
        cluster_props = self.dataset_df.groupby('Cluster').mean()
        print(cluster_props)
    
        
    def visualise_clusters(self):
        cluster_colors=sns.color_palette('mako', n_colors=self.n_clusters)  # use the 'mako' color scheme
        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids']
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
        legend_style=dict(borderpad=2, frameon=False, fontsize=8)
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axs.flatten()  # flatten the 2D array of AxesSubplot objects

        # Silhouette plot
        s_viz = SilhouetteVisualizer(self.model, colors='mako', ax=ax1)
        s_viz.fit(self.dataset)

        # Scatter plot of the cluster distributions
        ax2.scatter(
            self.dataset_df.iloc[:, 0],
            self.dataset_df.iloc[:, 1],
            c=self.dataset_df["Cluster"],
            cmap="mako",
            edgecolor='black'
        )
        ax2.scatter(
            self.model.cluster_centers_[:, 0],
            self.model.cluster_centers_[:, 1],
            s=200,
            c="red",
            edgecolor='black'
        )
        # Create a legend for the scatter plot
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) for label, color in zip(labels, cluster_colors)]
        legend2 = ax2.legend(handles=legend_elements, loc="best")
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
                                'frameon': False, 'fontsize':10})
        ax3.text(0.01, -0.09, '** 1 square ≈ 100 customers', weight = 'bold', style='italic', fontsize=8)
        
        ax4.axis('off')  # turn off the fourth subplot
        
        plt.suptitle('Credit Card Customer Clustering using K-Means\n', fontsize=14, **text_style)

        plt.savefig("Plots/clusters.png")  # save the plot as a .png file
        plt.close()  # close the plot


