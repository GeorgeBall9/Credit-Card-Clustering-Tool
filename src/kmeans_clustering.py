import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


class KMeansClustering:
    def __init__(self, dataset: pd.DataFrame, n_clusters: int = 4):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

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

    def fit_model(self):
        self.model.fit(self.dataset)
        self.dataset_df = pd.DataFrame(self.dataset)
        self.dataset_df["Cluster"] = self.model.labels_

    def silhouette_score(self):
        score = silhouette_score(self.dataset, self.model.labels_)
        print(f"Silhouette Score: {score}")

    def davies_bouldin_index(self):
        dbi = davies_bouldin_score(self.dataset, self.model.labels_)
        print(f"Davies-Bouldin Index: {dbi}")
        return dbi

    def visualise_clusters(self):
        plt.scatter(
            self.dataset_df.iloc[:, 0],
            self.dataset_df.iloc[:, 1],
            c=self.dataset_df["Cluster"],
            cmap="viridis",
        )
        plt.scatter(
            self.model.cluster_centers_[:, 0],
            self.model.cluster_centers_[:, 1],
            s=300,
            c="red",
        )
        plt.savefig("Plots/clusters.png")  # save the plot as a .png file
        plt.close()  # close the plot
