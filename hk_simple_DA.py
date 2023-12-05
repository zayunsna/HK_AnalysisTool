#!/usr/bin/env python
# -*- coding: utf-8 -*-
########
# This Library contains the basic tool for Feature Engineering of Data Analysis 
# Of course, all function already developed more famous package such as sklearn, pandas, numpy, scipy or so on..
# But, In order to study basic concept of each algorithms or methods, I built this function.
# And.. who knows, it might help with data analysis and make it faster :)
########

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
import seaborn as sns

def elastic_naming(form:str, n:int):
    names = []
    for i in range(n):
        name = form + '_' + str(i)
        names.append(name)
    return names

################################################################################################################################
#### This part is making the dataframe sample for the function test.
# Generating a dataset with 3 features
def make_testdf(n_samples:int, nFeatures:int):
    X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=nFeatures, random_state=42)
    column_names = elastic_naming(form='Feature', n=nFeatures)
    df = pd.DataFrame(X, columns=column_names)
    print(df.head())
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
################################################################################################################################

class PCA_Analysis:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.explained_variance = None
        self.cumulative_explained_variance = None
        self.pca_df = None
        self.principal_components = None

    def fit(self, dataset):
        data_scaled = self.scaler.fit_transform(dataset)
        self.principal_components = self.pca.fit_transform(data_scaled)

        # Explained variance ratio
        self.explained_variance = self.pca.explained_variance_ratio_
        print(f"Explained Variance Ratio: {self.explained_variance}")

        # Cumulative explained variance
        self.cumulative_explained_variance = np.cumsum(self.explained_variance)
        print(f"Cumulative Explained Variance: {self.cumulative_explained_variance}")

        # TODO : elastic_naming is temporary functions. It should be updated into the data based naming function.
        column_names = elastic_naming(form='Principal Component', n=self.n_components)
        self.pca_df = pd.DataFrame(data=self.principal_components, columns=column_names)

        return self

    def plot_pca(self):
        if self.pca_df is not None:
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(self.explained_variance) + 1), self.explained_variance, alpha=0.5, align='center', label='Individual explained variance')
            plt.step(range(1, len(self.cumulative_explained_variance) + 1), self.cumulative_explained_variance, where='mid', label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            plt.legend(loc='best')
            plt.title('Scree Plot')

            plt.subplot(1, 2, 2)
            plt.scatter(self.pca_df.iloc[:, 0], self.pca_df.iloc[:, 1])
            plt.xlabel(self.pca_df.columns[0])
            plt.ylabel(self.pca_df.columns[1])
            plt.title('2 component PCA')

            plt.tight_layout()
            plt.show()
        else:
            print("PCA not fitted yet.")

    def get_transformed_data(self):
        return self.pca_df

    def elastic_naming(self, form, n):
        return [f"{form} {i + 1}" for i in range(n)]


# == Test code of PCA
# df_scaled = make_testdf(n_samples=1000, nFeatures=5)
# pca = PCA_Analysis(n_components=3)
# pca.fit(df_scaled)
# pca.plot_pca()
# transformed_data = pca.get_transformed_data()
# print(transformed_data)


############### GMM (with K-Means)
from sklearn.cluster import KMeans
class GMM_Analysis:
    def __init__(self, n_components, n_iter, tol):
        # Initialize the Gaussian Mixture Model.
        # Parameters:
        # -> n_components (int): Number of Gaussian distributions (clusters).
        # -> n_iter (int): Maximum number of iterations for the EM algorithm.
        # -> tol (float): Tolerance for convergence.
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None
        self.converged = False

    def initialize_parameters(self, X):
        # Initialize the parameters using K-Means for initial guesses.

        # Parameters:
        # -> X (ndarray): Data points.

        # Using K-Means to initialize means
        kmeans = KMeans(n_clusters=self.n_components)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_

        # Initialize covariances and weights
        self.covariances = [np.cov(X.T) for _ in range(self.n_components)]
        self.weights = np.ones(self.n_components) / self.n_components

    def e_step(self, X):
        # E-step of the EM algorithm. Calculate responsibilities.
        # Parameters:
        # -> X (ndarray): Data points.

        # Returns:
        #   responsibilities (ndarray): Probability of each data point belonging to each cluster.
        responsibilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            rv = multivariate_normal(self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * rv.pdf(X)

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        # M-step of the EM algorithm. Update parameters based on responsibilities.
        # Parameters:
        # -> X (ndarray): Data points.
        # -> responsibilities (ndarray): Responsibilities from the E-step.
        Nk = responsibilities.sum(axis=0)

        # Update means
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]

        # Update weights
        self.weights = Nk / X.shape[0]

    def fit(self, X):
        # Fit the GMM to the data.
        # Parameters:
        # -> X (ndarray): Data points.
        self.initialize_parameters(X)

        log_likelihood_old = 0
        for _ in range(self.n_iter):
            # E-step
            responsibilities = self.e_step(X)

            # M-step
            self.m_step(X, responsibilities)

            # Check for convergence
            log_likelihood_new = np.sum(np.log(np.sum([k * multivariate_normal(self.means[k], self.covariances[k]).pdf(X) for k in range(self.n_components)], axis=0)))
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                self.converged = True
                break
            log_likelihood_old = log_likelihood_new

        return self

    def predict(self, X):
        # Predict the cluster for each data point.
        # Parameters:
        # -> X (ndarray): Data points.
        # Returns:
        # -> labels (ndarray): Cluster labels for each data point.
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def plot_gmm(self, X):
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], s=15)

        # Plot the Gaussian components
        for mean, covar in zip(self.means, self.covariances):
            # Calculate the eigenvalues and eigenvectors for the covariance matrix
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])

            # Calculate angle in degrees for the ellipse
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)  # Convert to degrees

            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, edgecolor='black')
            ell.set_clip_box(plt.gca().bbox)
            ell.set_alpha(0.5)
            plt.gca().add_artist(ell)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Gaussian Mixture Model')
        plt.show()

# == Test code of GMM
# X = make_testdf(n_samples=1000, nFeatures=5)
# gmm = GMM_Analysis(n_components=3, n_iter=100, tol=1e-3)
# gmm.fit(X)  # X is your data
# gmm.plot_gmm(X)
# labels = gmm.predict(X)
# print(labels)


############### Multiple Linear Regression

############### Lasso : If a specific feature is important than all, Lasso might gives better analysis result.

############### Ridge : If all feature is important uniformly, Ridge can gives better analysis performance than Lasso.