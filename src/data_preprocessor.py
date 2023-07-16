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
    
    def impute_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.fillna(dataset.mean(), inplace=True)
        return dataset
    
    def drop_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.drop(columns=['CUST_ID'])
        return dataset
    
    def normalise_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        normalised_data = scaler.fit_transform(dataset)
        normalised_dataset = pd.DataFrame(normalised_data, columns = dataset.columns)
        return normalised_dataset