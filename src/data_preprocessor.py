import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy import stats


class DataPreprocessor:
    def __init__(self, filepath):
        """DataPreprocessor class to load, preprocess, and analyse data.

        Args:
            filepath (str): The file path to the dataset.
        """
        if not os.path.isfile(filepath):
            raise ValueError(f"The provided file path {filepath} does not exist.")
        
        self.filepath = filepath 
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.

        """
        try:
            dataset = pd.read_csv(self.filepath)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.filepath}. Error: {str(e)}")  
          
        return dataset
    
    
    def plot_histograms(self, dataset: pd.DataFrame):
        """
        Plot histograms for each variable in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to plot histograms for.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset should be of type pd.DataFrame.")
        
        font = {'family' : 'serif'}
        plt.rc('font', **font)
        
        # Create a figure
        fig = plt.figure(figsize=(26,24))
        
        # Create histograms for each column in the dataset
        ax = dataset.hist(bins=50, figsize=(26, 24), color='aquamarine')

        # Increase font size of axis labels
        for axis in ax.flatten():
            axis.tick_params(axis='both', which='major', labelsize=16)
            axis.set_title(axis.get_title(), fontdict={'fontsize': 14, 'fontfamily': 'serif'})

        # Set the main title 
        plt.suptitle("Histogram representation of each variable", fontsize=28, fontweight='bold', fontfamily='serif')
        
        # Save the figure
        plt.savefig('Plots/PreProcessing/histograms.png')
        plt.close()
    
    def check_missing_values(self, dataset: pd.DataFrame):
        """
        Check for missing values in the dataset and plot a heatmap.

        Args:
            dataset (pd.DataFrame): The dataset to check for missing values.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset should be of type pd.DataFrame.")
        
        # Check for missing values in the dataset
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:\n", missing_values)
        
        # Get the list of columns with missing values and their counts
        columns_with_missing_values = missing_values[missing_values > 0]
        
        # Plot the missing values heatmap for columns with missing values
        if not columns_with_missing_values.empty:
            self.plot_missing_values_heatmap(dataset, columns_with_missing_values.index.tolist(), columns_with_missing_values.to_dict())
        
        
    def plot_missing_values_heatmap(self, dataset: pd.DataFrame, columns: list, missing_counts: dict):
        """
        Plot a heatmap of missing values in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to plot the heatmap for.
            columns (list): The list of columns with missing values.
            missing_counts (dict): The dictionary of columns with their corresponding missing value counts.

        """
        if not isinstance(dataset, pd.DataFrame) or not isinstance(columns, list) or not isinstance(missing_counts, dict):
            raise ValueError("Invalid input types. Expecting pd.DataFrame, list and dict for dataset, columns and missing_counts respectively.")
        
        # Select the columns with missing values
        missing_columns = dataset[columns]
        
        # Plot the missing values heatmap
        plt.figure(figsize=(22, 16))
        sns.heatmap(missing_columns.isnull(), cbar=False, cmap='Blues')
        
        ax = plt.gca()  # Get the current Axes instance
        ax.set_xticklabels(ax.get_xticklabels(), rotation=360, fontsize=14, fontfamily='serif')  # Rotate x-axis labels and increase font size
        plt.yticks(fontsize=10, fontfamily='serif')  # Increase y-axis label font size
        
        # Create a subtitle string based on the missing counts
        subtitle = ', '.join([f'{col}: {missing_counts[col]}' for col in columns])
        
        plt.suptitle('Heatmap of Missing Values within the Data', fontweight='bold', x=0.327, y=0.96, ha='center', fontsize=22, fontfamily='serif')
        plt.title(f'Missing values: {subtitle}', fontsize=16, fontstyle='italic', fontfamily='serif')
        
        plt.savefig('Plots/PreProcessing/missing_values_heatmap.png')
        plt.close()
    
    def drop_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the 'CUST_ID' column from the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to drop the column from.

        Returns:
            pd.DataFrame: The dataset with the column dropped.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset should be of type pd.DataFrame.")
        
        if 'CUST_ID' not in dataset.columns:
            raise ValueError("The column 'CUST_ID' does not exist in the dataset.")
        
        dataset = dataset.drop(columns=['CUST_ID'])
        return dataset
    
    def impute_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset with the column mean.

        Args:
            dataset (pd.DataFrame): The dataset to impute missing values.

        Returns:
            pd.DataFrame: The dataset with missing values imputed.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset should be of type pd.DataFrame.")
        
        dataset.fillna(dataset.mean(), inplace=True)
        
        # Check for missing values again
        missing_values = dataset.isnull().sum()
        print("Missing values in each column after imputation:\n", missing_values)

        
        return dataset
    
    def normalise_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Perform standardisation on the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to normalise.

        Returns:
            pd.DataFrame: The normalised dataset.

        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset should be of type pd.DataFrame.")
        
        try:
            scaler = StandardScaler()
            normalised_data = scaler.fit_transform(dataset)
        except Exception as e:
            raise ValueError(f"Failed to normalise data. Error: {str(e)}")
        
        normalised_dataset = pd.DataFrame(normalised_data, columns = dataset.columns)
        return normalised_dataset