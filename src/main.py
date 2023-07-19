import os
import time
from exploratory_data_analysis import EDA
from data_preprocessor import DataPreprocessor
from autoencoder_trainer import AutoencoderTrainer
from kmeans_clustering import KMeansClustering
from rfm_analysis import RFMAnalysis


def setup_directories():
    """
    Set up directories for storing output files and plots.
    """
    
    # List of plot directories to be created
    directories = ['Dataset', 'TrainedAutoencoder','Plots/EDA', 'Plots/PreProcessing', 'Plots/RFM', 'Plots/Clustering']

    # Loop over the directories
    for directory in directories:
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
    

def load_and_preprocess_data():
    """
    Load and preprocess the dataset.
    Returns:
        dataset (pandas.DataFrame): The preprocessed dataset.
        normalised_dataset (pandas.DataFrame): The normalised dataset.
    """
    
    # Create an instance of DataPreprocessor
    data_preprocessor = DataPreprocessor(filepath='Dataset/data.csv')

    dataset = data_preprocessor.load_data()
    print("Initial dataset:")
    print(dataset.head())

    # Histogram plot
    data_preprocessor.plot_histograms(dataset)

    # Check for missing values
    data_preprocessor.check_missing_values(dataset)

    # Drop the CUST_ID column
    dataset = data_preprocessor.drop_columns(dataset)


    print("\nDataFrame after dropping columns:")
    print(dataset.head())

    dataset = data_preprocessor.impute_missing_values(dataset)

    normalised_dataset = data_preprocessor.normalise_data(dataset)
    print("\nNormalised dataset:")
    print(normalised_dataset.head())
    
    return dataset, normalised_dataset

def perform_eda(dataset):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    Args:
        dataset (pandas.DataFrame): The dataset to perform EDA on.
    """
    
    # Create an instance of EDA and perform EDA
    eda = EDA(dataset)
    eda.plot_correlation_heatmap()
    eda.plot_credit_limit_vs_balance()
    eda.plot_credit_limit_vs_installment()
    eda.plot_purchases_vs_tenure()


def perform_rfm_analysis(dataset):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis on the dataset.
    Args:
        dataset (pandas.DataFrame): The dataset to perform RFM analysis on.
    """
    
    # Step : Perform RFM Analysis
    rfm_analysis = RFMAnalysis(dataset)
    rfm_values = rfm_analysis.calculate_rfm_values()
    rfm_analysis.plot_customer_segments(rfm_values)
    print("\n RFM analysis...")
    print(rfm_values)  # print RFM values


def train_autoencoder(normalised_dataset):
    """
    Train the autoencoder for dimensionality reduction.
    Args:
        normalised_dataset (pandas.DataFrame): The normalised dataset.
    Returns:
        autoencoder_trainer (AutoencoderTrainer): The trained autoencoder trainer.
    """
    
    # Step 2: Train the Autoencoder for dimensionality reduction
    autoencoder_trainer = AutoencoderTrainer(normalised_dataset)

    # Check if a trained model already exists
    if os.path.exists('TrainedAutoencoder/best_model.h5'):
        # If it does, load the model
        autoencoder_trainer.load_model('TrainedAutoencoder/best_model.h5')
    else:
        # If it doesn't, build and train a new model
        autoencoder = autoencoder_trainer.build_autoencoder()
        print(autoencoder.summary())
        print("\nStarting training...")
        trained_autoencoder = autoencoder_trainer.train_autoencoder(autoencoder)
        print("\nTraining complete.")
        
    autoencoder_trainer.reconstruction_error()
    
    return autoencoder_trainer

def encode_data(autoencoder_trainer):
    """
    Use the autoencoder to encode the data.
    Args:
        autoencoder_trainer (AutoencoderTrainer): The trained autoencoder trainer object.
    Returns:
        encoded_data (numpy.ndarray): The encoded data.
    """
    
    # Use the autoencoder to encode the data
    encoded_data = autoencoder_trainer.encode_data(autoencoder_trainer.autoencoder)
    print("\nShape of encoded data:", encoded_data.shape)
    
    return encoded_data

def apply_kmeans_clustering(encoded_data):
    """
    Apply K-means clustering on the encoded data.
    Args:
        encoded_data (numpy.ndarray): The encoded data.
    """
    
    # Step 3: Apply K-means clustering
    kmeans_clustering = KMeansClustering(encoded_data)

    kmeans_clustering.inertia_plot()
    kmeans_clustering.silhouette_analysis()
    kmeans_clustering.fit_model()
    kmeans_clustering.silhouette_score()
    kmeans_clustering.calinski_harabasz_index()
    kmeans_clustering.davies_bouldin_index() 
    kmeans_clustering.cluster_properties()
    kmeans_clustering.visualise_clusters()
    print("\n",kmeans_clustering.cluster_summary().to_string())



def main():
    
    start_time = time.time() # Record the start time
    
    setup_directories()
    
    dataset, normalised_dataset = load_and_preprocess_data()
    
    perform_eda(dataset)
    
    perform_rfm_analysis(dataset)
    
    autoencoder_trainer = train_autoencoder(normalised_dataset)
    
    encoded_data = encode_data(autoencoder_trainer)

    apply_kmeans_clustering(encoded_data)

    end_time = time.time()  # Save the current time at the end of your script

    runtime = end_time - start_time  # Calculate the runtime

    print(f"\nThe runtime of the script is {runtime} seconds.")
    
if __name__ == "__main__":
    main()
