import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath 
        
    def load_data(self) -> pd.DataFrame:
        #Load the dataset from the CSV file
        dataset = pd.read_csv(self.filepath)    
        return dataset
    
    # Histogram plot of data variables for intial look at dataset
    def plot_histograms(self, dataset: pd.DataFrame):
        # Set the font
        font = {'family' : 'serif'}
        plt.rc('font', **font)
        
        # Create a figure
        fig = plt.figure(figsize=(26,24))
        
        # Create histograms for each column in the dataset
        dataset.hist(bins=50, figsize=(26,24), color='aquamarine')
        
        # Set the main title 
        plt.suptitle("Histogram representation of each variable", fontsize=22, fontweight='bold', fontfamily='serif')
        
        # Save the figure
        plt.savefig('Plots/PreProcessing/histograms.png')
        plt.close()
    
    def check_missing_values(self, dataset: pd.DataFrame):
        # Check for missing values in the dataset
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:\n", missing_values)
        
        # Get the list of columns with missing values and their counts
        columns_with_missing_values = missing_values[missing_values > 0]
        
        # Plot the missing values heatmap for columns with missing values
        if not columns_with_missing_values.empty:
            self.plot_missing_values_heatmap(dataset, columns_with_missing_values.index.tolist(), columns_with_missing_values.to_dict())
        
        
    def plot_missing_values_heatmap(self, dataset: pd.DataFrame, columns: list, missing_counts: dict):
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
        dataset = dataset.drop(columns=['CUST_ID'])
        return dataset
    
    def impute_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.fillna(dataset.mean(), inplace=True)
        
        # Check for missing values again
        missing_values = dataset.isnull().sum()
        print("Missing values in each column after imputation:\n", missing_values)

        
        return dataset
    
    def normalise_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        normalised_data = scaler.fit_transform(dataset)
        normalised_dataset = pd.DataFrame(normalised_data, columns = dataset.columns)
        return normalised_dataset