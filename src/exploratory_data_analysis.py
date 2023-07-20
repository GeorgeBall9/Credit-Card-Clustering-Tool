import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

class EDA:
    def __init__(self, dataset: pd.DataFrame):
        """
        Exploratory Data Analysis (EDA) class to perform data exploration and visualization.

        Args:
            dataset (pd.DataFrame): The dataset to be analyzed.

        Attributes:
            dataset (pd.DataFrame): The dataset to be analyzed.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("The provided dataset must be a pandas DataFrame.")
        
        self.dataset = dataset

    def plot_correlation_heatmap(self):
        """
        Plot a correlation heatmap for numerical variables in the dataset.

        """
        
        corr = self.dataset.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Create a mask for the upper triangle

        fig, ax = plt.subplots(figsize=(24, 18))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='mako', linewidths=0.1, cbar=False, annot_kws={"size":18})
        
        yticks, ylabels = plt.yticks()
        xticks, xlabels = plt.xticks()
        ax.set_xticklabels(xlabels, size=14, fontfamily='serif')
        ax.set_yticklabels(ylabels, size=14, fontfamily='serif')
        
        plt.suptitle('Correlation Map of Numerical Variables', fontweight='bold', x=0.327, y=0.96, ha='center', fontsize=18, fontfamily='serif')
        plt.title('Some variables have significant correlations with other variables (> 0.5).\n', fontsize=16, loc='right', fontfamily='serif')
        plt.tight_layout(rect=[0, 0.04, 1, 1.01])
        
        plt.savefig('Plots/EDA/heatmap_correlation.png')  # save the plot as a .png file
        plt.close()  # close the plot


    def plot_credit_limit_vs_balance(self):
        """
        Plot a scatter plot comparing Credit Limit vs. Balance based on Tenure.

        """
        # Create a figure and a grid of subplots
        fig = plt.figure(figsize=(22, 14))
        unique_tenures = sorted(self.dataset['TENURE'].unique())
        gs = GridSpec(len(unique_tenures) + 1, 2, width_ratios=[3, 1], figure=fig)

        # Define a colour palette
        color_palette = sns.color_palette("mako", len(unique_tenures))

        # Main scatter plot comparing the CREDIT_LIMIT vs BALANCE based on TENURE
        ax_main = fig.add_subplot(gs[:, 0])
        sns.scatterplot(data=self.dataset, x='CREDIT_LIMIT', y='BALANCE', hue='TENURE', palette=color_palette, ax=ax_main, edgecolor=None)
        ax_main.set_title('Credit Limit vs. Balance based on Tenure', fontweight='bold', fontsize=20, fontfamily='serif')
        
        # Adjust font of x and y ticks
        yticks, ylabels = plt.yticks()
        xticks, xlabels = plt.xticks()
        ax_main.set_xticklabels(xlabels, size=14)
        ax_main.set_yticklabels(ylabels, size=14)
        
        # Make x and y axis labels bigger and bold
        ax_main.xaxis.label.set_fontsize(16)
        ax_main.yaxis.label.set_fontsize(16)
        ax_main.xaxis.label.set_fontweight('bold')
        ax_main.yaxis.label.set_fontweight('bold')

        # Subplots for each unique value of TENURE
        for i, tenure in enumerate(unique_tenures):
            ax_sub = fig.add_subplot(gs[i, 1])
            subset = self.dataset[self.dataset['TENURE'] == tenure]
            other = self.dataset[self.dataset['TENURE'] != tenure]
            sns.scatterplot(data=other, x='CREDIT_LIMIT', y='BALANCE', ax=ax_sub, color='lightgrey', alpha=0.2, edgecolor=None)
            sns.scatterplot(data=subset, x='CREDIT_LIMIT', y='BALANCE', ax=ax_sub, label=f'TENURE: {tenure}', color=color_palette[i], edgecolor="black")
            ax_sub.set_xlabel('')
            ax_sub.set_ylabel('')
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])

        # Adjust the spacing between the subplots
        plt.subplots_adjust(hspace=0.5)

        # Save the plot as a .png file
        plt.savefig('Plots/EDA/credit_limit_vs_balance.png')
        plt.close()  # close the plot
        
    def plot_credit_limit_vs_installment(self):
        """
        Plot a scatter plot comparing Credit Limit vs. Installment Purchases based on Tenure.

        """
        # Create a figure and a grid of subplots
        fig = plt.figure(figsize=(22, 14))
        unique_tenures = sorted(self.dataset['TENURE'].unique())
        gs = GridSpec(len(unique_tenures) + 1, 2, width_ratios=[3, 1], figure=fig)

        # Define a colour palette
        color_palette = sns.color_palette("mako", len(unique_tenures))

        # Main scatter plot comparing the CREDIT_LIMIT vs BALANCE based on TENURE
        ax_main = fig.add_subplot(gs[:, 0])
        sns.scatterplot(data=self.dataset, x='CREDIT_LIMIT', y='INSTALLMENTS_PURCHASES', hue='TENURE', palette=color_palette, ax=ax_main, edgecolor=None)
        ax_main.set_title('Credit Limit vs. Installments Purchases based on Tenure', fontweight='bold', fontsize=20, fontfamily='serif')
        
        # Adjust font of x and y ticks
        yticks, ylabels = plt.yticks()
        xticks, xlabels = plt.xticks()
        ax_main.set_xticklabels(xlabels, size=14)
        ax_main.set_yticklabels(ylabels, size=14)
        
        # Make x and y axis labels bigger and bold
        ax_main.xaxis.label.set_fontsize(16)
        ax_main.yaxis.label.set_fontsize(16)
        ax_main.xaxis.label.set_fontweight('bold')
        ax_main.yaxis.label.set_fontweight('bold')

        # Subplots for each unique value of TENURE
        for i, tenure in enumerate(unique_tenures):
            ax_sub = fig.add_subplot(gs[i, 1])
            subset = self.dataset[self.dataset['TENURE'] == tenure]
            other = self.dataset[self.dataset['TENURE'] != tenure]
            sns.scatterplot(data=other, x='CREDIT_LIMIT', y='INSTALLMENTS_PURCHASES', ax=ax_sub, color='lightgrey', alpha=0.2, edgecolor=None)
            sns.scatterplot(data=subset, x='CREDIT_LIMIT', y='INSTALLMENTS_PURCHASES', ax=ax_sub, label=f'TENURE: {tenure}', color=color_palette[i], edgecolor="black")
            ax_sub.set_xlabel('')
            ax_sub.set_ylabel('')
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])

        # Adjust the spacing between the subplots
        plt.subplots_adjust(hspace=0.5)

        # Save the plot as a .png file
        plt.savefig('Plots/EDA/credit_limit_vs_installments_purchases.png')
        plt.close()  # close the plot

    def plot_purchases_vs_tenure(self):
        """
        Plot the relationship between Purchases Amount and Total Purchase Transaction based on Tenure.

        """
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
        axes[0].set_title('Customer Purchases Amount', fontweight='bold', fontstyle='italic', fontsize=12, fontfamily='serif')
        axes[0].set_xlabel('PURCHASES', fontweight='bold', fontfamily='serif')
        axes[0].set_ylabel('TENURE', fontweight='bold', fontfamily='serif')

        # Add mean and max labels
        for i in range(len(purchases_data)):
            axes[0].text(purchases_data.loc[i, 'mean'], purchases_data.loc[i, 'TENURE'] + 0.15, f"{purchases_data.loc[i, 'mean']:.2f}", va='center', ha='center', color=color_palette[1], fontsize=8, fontweight='bold', fontfamily='serif')
            axes[0].text(purchases_data.loc[i, 'max'], purchases_data.loc[i, 'TENURE'] + 0.15, f"{purchases_data.loc[i, 'max']:.2f}", va='center', ha='center', color=color_palette[2], fontsize=8, fontweight='bold', fontfamily='serif')

        # Plot PURCHASES_TRX vs TENURE
        axes[1].hlines(purchases_trx_data['TENURE'], purchases_trx_data['min'], purchases_trx_data['max'], color='black')
        axes[1].plot(purchases_trx_data['min'], purchases_trx_data['TENURE'], 'o', color=color_palette[0], markersize=10, label='Min')
        axes[1].plot(purchases_trx_data['mean'], purchases_trx_data['TENURE'], 'o', color=color_palette[1], markersize=10, label='Mean')
        axes[1].plot(purchases_trx_data['max'], purchases_trx_data['TENURE'], 'o', color=color_palette[2], markersize=10, label='Max')
        axes[1].set_title('Total Purchase Transcations', fontweight='bold', fontstyle='italic', fontsize=12, fontfamily='serif')
        axes[1].set_xlabel('PURCHASES_TRX', fontweight='bold', fontfamily='serif')
        axes[1].set_ylabel('TENURE', fontweight='bold', fontfamily='serif')

        # Add mean and max labels
        for i in range(len(purchases_trx_data)):
            axes[1].text(purchases_trx_data.loc[i, 'mean'], purchases_trx_data.loc[i, 'TENURE'] + 0.2, f"{purchases_trx_data.loc[i, 'mean']:.2f}", va='top', ha='center', color=color_palette[1], fontsize=8, fontweight='bold', fontfamily='serif')
            axes[1].text(purchases_trx_data.loc[i, 'max'], purchases_trx_data.loc[i, 'TENURE'] + 0.2, f"{purchases_trx_data.loc[i, 'max']:.2f}", va='top', ha='center', color=color_palette[2], fontsize=8, fontweight='bold', fontfamily='serif')

        # Set the overall title and legend
        plt.suptitle('Purchases Amount and Total Purchase Transaction vs. Tenure Comparison', fontsize=14, fontweight='bold', fontfamily='serif')
        axes[0].legend()
        axes[1].legend()
        
        # Add padding to x-axis limits and vertical dashed-lines to x-axis ticks
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)

        # Save the plot as a .png file
        plt.savefig('Plots/EDA/purchases_vs_tenure.png')
        plt.close()  # close the plot


