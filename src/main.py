import os
from exploratory_data_analysis import EDA
from data_preprocessor import DataPreprocessor
from autoencoder_trainer import AutoencoderTrainer
from kmeans_clustering import KMeansClustering
from rfm_analysis import RFMAnalysis

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

# Drop the CUST_ID column
dataset = data_preprocessor.drop_columns(dataset)

# Create an instance of EDA and perform EDA
eda = EDA(dataset)
eda.check_missing_values()
eda.plot_correlation_heatmap()
eda.plot_pairplot()
eda.plot_credit_limit_vs_balance()

null_values = dataset.isnull().sum()
null_values = null_values[null_values > 0]
print("\nNull values in dataset:")
print(null_values)


print("\nDataset after dropping columns:")
print(dataset.head())

dataset = data_preprocessor.impute_missing_values(dataset)
print("\nDataset after imputing missing values:")
print(dataset.head())

normalised_dataset = data_preprocessor.normalise_data(dataset)
print("\nNormalised dataset:")
print(normalised_dataset.head())

# Plot correlation heatmap
data_preprocessor.plot_correlation_heatmap(dataset)

# Step : Perform RFM Analysis
rfm_analysis = RFMAnalysis(dataset)
rfm_values = rfm_analysis.calculate_rfm_values()
print("\n RFM analysis...")
print(rfm_values)  # print RFM values

# Step 2: Train the Autoencoder for dimensionality reduction
autoencoder_trainer = AutoencoderTrainer(normalised_dataset)

# Check if a trained model already exists
if os.path.exists('TrainedAutoencoder/trained_autoencoder.h5'):
    # If it does, load the model
    autoencoder_trainer.load_model('TrainedAutoencoder/trained_autoencoder.h5')
else:
    # If it doesn't, build and train a new model
    autoencoder = autoencoder_trainer.build_autoencoder()
    print(autoencoder.summary())
    print("Starting training...")
    trained_autoencoder = autoencoder_trainer.train_autoencoder(autoencoder)
    print("Training complete.")
    autoencoder_trainer.save_model('TrainedAutoencoder/trained_autoencoder.h5')  # Save the model to a file

# Use the autoencoder to encode the data
encoded_data = autoencoder_trainer.encode_data(autoencoder_trainer.autoencoder)
print("Shape of encoded data:", encoded_data.shape)


# Step 3: Apply K-means clustering
kmeans_clustering = KMeansClustering(encoded_data)

kmeans_clustering.inertia_plot()
kmeans_clustering.fit_model()
kmeans_clustering.silhouette_score()
kmeans_clustering.davies_bouldin_index() 
kmeans_clustering.visualise_clusters()

