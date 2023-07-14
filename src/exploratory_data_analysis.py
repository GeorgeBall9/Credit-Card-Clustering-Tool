import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class EDA:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def check_missing_values(self):
        # Check for missing values in the dataset
        missing_values = self.dataset.isnull().sum()
        print("Missing values in each column:\n", missing_values)

    def plot_correlation_heatmap(self):
        # Plot a correlation heatmap
        corr = self.dataset.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.savefig('Plots/correlation.png')  # save the plot as a .png file
        plt.close()  # close the plot

    def plot_pairplot(self):
        # Plot a pairplot to visualise the pairwise relationships between different features
        sns.pairplot(self.dataset)
        plt.savefig('Plots/pairplot.png')  # save the plot as a .png file
        plt.close()  # close the plot

    def plot_credit_limit_vs_balance(self):
        # Create a figure and a grid of subplots
        fig = plt.figure(figsize=(22, 14))
        unique_tenures = sorted(self.dataset['TENURE'].unique())
        gs = GridSpec(len(unique_tenures) + 1, 2, width_ratios=[3, 1], figure=fig)

        # Define a color palette
        color_palette = sns.color_palette("mako", len(unique_tenures))

        # Main scatter plot comparing the CREDIT_LIMIT vs BALANCE based on TENURE
        ax_main = fig.add_subplot(gs[:, 0])
        sns.scatterplot(data=self.dataset, x='CREDIT_LIMIT', y='BALANCE', hue='TENURE', palette=color_palette, ax=ax_main, edgecolor=None)
        ax_main.set_title('CREDIT_LIMIT vs BALANCE based on TENURE')

        # Subplots for each unique value of TENURE
        for i, tenure in enumerate(unique_tenures):
            ax_sub = fig.add_subplot(gs[i, 1])
            subset = self.dataset[self.dataset['TENURE'] == tenure]
            other = self.dataset[self.dataset['TENURE'] != tenure]
            sns.scatterplot(data=other, x='CREDIT_LIMIT', y='BALANCE', ax=ax_sub, color='lightgrey', alpha=0.2, edgecolor=None)
            sns.scatterplot(data=subset, x='CREDIT_LIMIT', y='BALANCE', ax=ax_sub, label=f'TENURE: {tenure}', color=color_palette[i], edgecolor="black")
            ax_sub.set_title(f'TENURE: {tenure}', fontsize=8)  # Adjust the font size here
            ax_sub.set_xlabel('')
            ax_sub.set_ylabel('')
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])

        # Adjust the spacing between the subplots
        plt.subplots_adjust(hspace=0.5)

        # Save the plot as a .png file
        plt.savefig('Plots/credit_limit_vs_balance.png')
        plt.close()  # close the plot



