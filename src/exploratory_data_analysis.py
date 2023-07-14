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

    def plot_purchases_vs_tenure(self):
        # Prepare the data
        purchases_data = self.dataset.groupby('TENURE')['PURCHASES'].agg(['min', 'mean', 'max']).reset_index()
        purchases_trx_data = self.dataset.groupby('TENURE')['PURCHASES_TRX'].agg(['min', 'mean', 'max']).reset_index()

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Define a color palette
        color_palette = sns.color_palette("mako", 3)

        # Plot PURCHASES vs TENURE
        axes[0].hlines(purchases_data['TENURE'], purchases_data['min'], purchases_data['max'], color='black')
        axes[0].plot(purchases_data['min'], purchases_data['TENURE'], 'o', color=color_palette[0], markersize=10, label='Min')
        axes[0].plot(purchases_data['mean'], purchases_data['TENURE'], 'o', color=color_palette[1], markersize=10, label='Mean')
        axes[0].plot(purchases_data['max'], purchases_data['TENURE'], 'o', color=color_palette[2], markersize=10, label='Max')
        axes[0].set_title('Customer Purchases Amount', fontweight='bold')
        axes[0].set_xlabel('PURCHASES', fontweight='bold')
        axes[0].set_ylabel('TENURE', fontweight='bold')

        # Add mean and max labels
        for i in range(len(purchases_data)):
            axes[0].text(purchases_data.loc[i, 'mean'], purchases_data.loc[i, 'TENURE'] + 0.2, f"{purchases_data.loc[i, 'mean']:.2f}", va='center', ha='center', color=color_palette[1], fontsize=8, fontweight='bold')
            axes[0].text(purchases_data.loc[i, 'max'], purchases_data.loc[i, 'TENURE'] + 0.2, f"{purchases_data.loc[i, 'max']:.2f}", va='center', ha='center', color=color_palette[2], fontsize=8, fontweight='bold')

        # Plot PURCHASES_TRX vs TENURE
        axes[1].hlines(purchases_trx_data['TENURE'], purchases_trx_data['min'], purchases_trx_data['max'], color='black')
        axes[1].plot(purchases_trx_data['min'], purchases_trx_data['TENURE'], 'o', color=color_palette[0], markersize=10, label='Min')
        axes[1].plot(purchases_trx_data['mean'], purchases_trx_data['TENURE'], 'o', color=color_palette[1], markersize=10, label='Mean')
        axes[1].plot(purchases_trx_data['max'], purchases_trx_data['TENURE'], 'o', color=color_palette[2], markersize=10, label='Max')
        axes[1].set_title('Total Purchase Transcations', fontweight='bold')
        axes[1].set_xlabel('PURCHASES_TRX', fontweight='bold')
        axes[1].set_ylabel('TENURE', fontweight='bold')

        # Add mean and max labels
        for i in range(len(purchases_trx_data)):
            axes[1].text(purchases_trx_data.loc[i, 'mean'], purchases_trx_data.loc[i, 'TENURE'] + 0.2, f"{purchases_trx_data.loc[i, 'mean']:.2f}", va='top', ha='center', color=color_palette[1], fontsize=8, fontweight='bold')
            axes[1].text(purchases_trx_data.loc[i, 'max'], purchases_trx_data.loc[i, 'TENURE'] + 0.2, f"{purchases_trx_data.loc[i, 'max']:.2f}", va='top', ha='center', color=color_palette[2], fontsize=8, fontweight='bold')

        # Set the overall title and legend
        plt.suptitle('Purchases Amount and Total Purchase Transaction Comparison', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[1].legend()
        
        # Add padding to x-axis limits and vertical dashed-lines to x-axis ticks
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)

        # Save the plot as a .png file
        plt.savefig('Plots/purchases_vs_tenure.png')
        plt.close()  # close the plot


