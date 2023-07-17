import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



class RFMAnalysis:
    def __init__(self, dataset: pd.DataFrame):
        """
        RFM Analysis class to perform RFM analysis on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to be used for RFM analysis.

        Attributes:
            dataset (pd.DataFrame): The dataset to be used for RFM analysis.

        """
        self.dataset = dataset

    def calculate_recency(self):
        """
        Calculate the recency values for RFM analysis.

        Returns:
            pd.DataFrame: The recency values.

        """
        recency = pd.DataFrame(index=self.dataset.index)
        recency['Recency'] = 10  # arbitrary recency value
        return recency

    def calculate_frequency(self):
        """
        Calculate the frequency values for RFM analysis.

        Returns:
            pd.DataFrame: The frequency values.

        """
        frequency_columns = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']
        frequency = self.dataset[frequency_columns].sum(axis=1).to_frame('Frequency')
        return frequency

    def calculate_monetary(self):
        """
        Calculate the monetary values for RFM analysis.

        Returns:
            pd.DataFrame: The monetary values.

        """
        monetary_columns = ['PURCHASES', 'PAYMENTS']
        monetary = self.dataset[monetary_columns].sum(axis=1).to_frame('Monetary')
        return monetary

    def calculate_rfm_values(self):
        """
        Calculate the RFM values based on the recency, frequency, and monetary values.

        Returns:
            pd.DataFrame: The RFM values.

        """
        # Call Recency, Frequency and Monetary values
        recency = self.calculate_recency()
        frequency = self.calculate_frequency()
        monetary = self.calculate_monetary()

        # Merge dataframes
        rfm = pd.concat([recency, frequency, monetary], axis=1)

        # Calculate ranks
        rfm['R_rank'] = rfm['Recency'].rank(ascending=False)
        rfm['F_rank'] = rfm['Frequency'].rank(ascending=True)
        rfm['M_rank'] = rfm['Monetary'].rank(ascending=True)
        
        # Normalize ranks
        rfm['R_rank_norm'] = (rfm['R_rank']) / (rfm['R_rank'].max()) * 100
        rfm['F_rank_norm'] = (rfm['F_rank']) / (rfm['F_rank'].max()) * 100
        rfm['M_rank_norm'] = (rfm['M_rank']) / (rfm['M_rank'].max()) * 100

        # Drop the original rank columns
        rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

        # Calculate RFM score
        rfm['RFMScore'] = 0.15 * rfm['R_rank_norm'] + 0.28 * rfm['F_rank_norm'] + 0.57 * rfm['M_rank_norm']

        # Scale RFM score to a scale of 0 to 5
        rfm['RFMScore'] *= 0.05

        # Assign customers to segments
        rfm["Customer Segment"] = pd.cut(rfm['RFMScore'], bins=[0, 1.6, 3, 4, 4.5, 5], labels=['Lost Customer', 'Low Value Customer', 'Medium Value Customer', 'High Value Customer', 'Top Customer'], include_lowest=True)

        return rfm
    
    
    def plot_customer_segments(self, rfm):
        """
        Plot the customer segments based on the RFM analysis.

        Args:
            rfm (pd.DataFrame): The RFM values.

        """
        segment_counts = rfm['Customer Segment'].value_counts()
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # specify colour scheme
        # Create a color palette with the same number of colors as segments
        colors = sns.color_palette('mako', n_colors=segment_counts.shape[0] + 1) # Add extra colour to then skip first later (Too dark)
       
        plt.figure(figsize=(14, 8))
        patches, texts, autotexts = plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', colors=colors[1:])
        
        for autotext in autotexts:
            autotext.set_fontsize(14)
            autotext.set_fontfamily('serif')
        
        plt.legend(patches, segment_counts.index, loc="center left", bbox_to_anchor=(1.2, 0.5))
        plt.title('Customer Segments from RFM Analysis', fontsize=18, fontweight='bold', fontfamily='serif')
        plt.savefig("Plots/RFM/rfm_segments.png")  # save the plot as a .png file
        plt.close()  # close the plot

