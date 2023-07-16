import os
from exploratory_data_analysis import EDA
from data_preprocessor import DataPreprocessor
from autoencoder_trainer import AutoencoderTrainer
from kmeans_clustering import KMeansClustering
from rfm_analysis import RFMAnalysis

import time

start_time = time.time()

# Check if the Plots directory exists, if not, create it
if not os.path.exists('Plots'):
    os.makedirs('Plots')
    
# Check if the TrainedAutoencoder directory exists, if not, create it
if not os.path.exists('TrainedAutoencoder'):
    os.makedirs('TrainedAutoencoder')

# Step 1: Preprocess the data
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


print("\nDataset after dropping columns:")
print('.: Dataframe after Dropping Variables :.')
print(dataset.head())

dataset = data_preprocessor.impute_missing_values(dataset)

normalised_dataset = data_preprocessor.normalise_data(dataset)
print("\nNormalised dataset:")
print(normalised_dataset.head())


# Create an instance of EDA and perform EDA
eda = EDA(dataset)
eda.plot_correlation_heatmap()
eda.plot_credit_limit_vs_balance()
eda.plot_credit_limit_vs_installment()
eda.plot_purchases_vs_tenure()


# Step : Perform RFM Analysis
rfm_analysis = RFMAnalysis(dataset)
rfm_values = rfm_analysis.calculate_rfm_values()
print("\n RFM analysis...")
print(rfm_values)  # print RFM values

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
    print("Starting training...")
    trained_autoencoder = autoencoder_trainer.train_autoencoder(autoencoder)
    print("Training complete.")
    
autoencoder_trainer.reconstruction_error()

# Use the autoencoder to encode the data
encoded_data = autoencoder_trainer.encode_data(autoencoder_trainer.autoencoder)
print("Shape of encoded data:", encoded_data.shape)


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


end_time = time.time()  # Save the current time at the end of your script

runtime = end_time - start_time  # Calculate the runtime

print(f"The runtime of the script is {runtime} seconds.")

