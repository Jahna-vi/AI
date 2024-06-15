# AI
Cit internship studio
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Load the face dataset
face_db = np.load('face_dataset.npy')

# Calculate the mean of each observation
mean_face = np.mean(face_db, axis=1)

# Do mean Zero
mean_zero_faces = face_db - mean_face[:, np.newaxis]

# Calculate Co-Variance of the Mean aligned faces
cov_matrix = np.cov(mean_zero_faces)

# Do eigenvalue and eigenvector decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Find the best direction (Generation of feature vectors)
k = 50  # number of selected eigenvectors
feature_vector = eigenvectors[:, :k]

# Generating Eigenfaces
eigenfaces = np.dot(mean_zero_faces.T, feature_vector)

# Generate Signature of Each Face
signatures = np.dot(eigenfaces.T, mean_zero_faces)

# Apply ANN for training
ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
ann.fit(signatures.T, np.arange(signatures.shape[1]))

# Testing
test_image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
test_image = test_image.reshape(-1, 1)

# Do mean Zero
test_image = test_image - mean_face[:, np.newaxis]

# Project this mean aligned face to eigenfaces
projected_test_face = np.dot(eigenfaces.T, test_image)

# Use the trained ANN model to predict the unknown face
predicted_face = ann.predict(projected_test_face.T)

print("Predicted Face:", predicted_face)
