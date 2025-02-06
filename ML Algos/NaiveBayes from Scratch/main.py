import numpy as np
import pandas as pd

# Gaussian likelihood calculator function
def likelihood_calculator(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Function to calculate mean and standard deviation for each feature by class
def calculate_class_stats(data, target_column):
    stats = {}
    # Separate data by class
    for target_value in data[target_column].unique():
        class_data = data[data[target_column] == target_value]
        class_stats = {
            'means': class_data.drop(target_column, axis=1).mean().values,
            'stds': class_data.drop(target_column, axis=1).std().values
        }
        stats[target_value] = class_stats
    return stats

# Function to calculate the prior probability for each class
def calculate_priors(data, target_column):
    priors = {}
    total_count = len(data)
    for target_value in data[target_column].unique():
        class_count = len(data[data[target_column] == target_value])
        priors[target_value] = class_count / total_count
    return priors

# Gaussian Naive Bayes prediction function
def predict(stats, priors, sample):
    probabilities = {}
    # Calculate probability for each class
    for target_class, class_stats in stats.items():
        probabilities[target_class] = priors[target_class]
        means = class_stats['means']
        stds = class_stats['stds']
        # Multiply likelihoods for each feature
        for i in range(len(sample)):
            probabilities[target_class] *= likelihood_calculator(sample[i], means[i], stds[i])
    # Select the class with the highest probability
    return max(probabilities, key=probabilities.get)

# Loading and preparing data
# Example (replace this with your own data)
data = pd.DataFrame({
    'feature1': [5.1, 4.9, 4.7, 6.2, 5.9, 6.0],
    'feature2': [3.5, 3.0, 3.2, 2.9, 3.1, 3.2],
    'feature3': [1.4, 1.5, 1.3, 4.7, 5.1, 4.9],
    'diabetes': [0, 0, 0, 1, 1, 1]
})

# Initialize sample to predict
test_sample = [5.0, 3.2, 1.5]

# Calculate class statistics and priors
target_column = 'diabetes'
class_stats = calculate_class_stats(data, target_column)
priors = calculate_priors(data, target_column)

# Predict class
prediction = predict(class_stats, priors, test_sample)
print(f"The predicted class for the sample {test_sample} is: {prediction}")
