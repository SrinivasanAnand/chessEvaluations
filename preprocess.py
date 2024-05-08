import pandas as pd
import chess
from sklearn.linear_model import LogisticRegression

import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings(action='ignore')
np.random.seed = 1

def run_first_version():
    # Load the dataset
    df = pd.read_csv('toyChessData.csv')  # Replace 'your_dataset.csv' with your actual file name

    # white is capital and positive, black is lowercase and negative
    key = {'r': 0, 'n': 1, 'b': 2, 'q': 3,'k':4, 'p': 5,'R': 6, 'N': 7, 'B': 8, 'Q': 9,'K': 10, 'P': 11}

    x = []
    y = []

    # Transforming
    for idx, row in df.iterrows():
        board = chess.Board(row['FEN'])

        bitboard = np.zeros((12, 64), dtype=np.int8)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                bitboard[key[piece.symbol()]][square] = 1

        features = [np.packbits(bitboard[i]).view(np.uint64)[0] for i in range(12)]
        features.append(int(board.turn))
        features.append(board.castling_rights)

        x.append(features)

        eval = row['Evaluation'].strip('+')
        if '#' in eval:
            y.append(np.sign(int(eval[1:])) * 1000)
        else:
            y.append(int(eval))

    x = np.array(x)
    y = np.array(y)

    y[y > 0] = 1 # white wins, pos
    y[y <= 0] = 0 # black wins, neg

    shuffled_indices = np.random.permutation(x.shape[0])

    # Choose the first 60% as training set, next 20% as validation and the rest as testing
    train_split_idx = int(0.60 * x.shape[0])
    val_split_idx = int(0.80 * x.shape[0])

    train_indices = shuffled_indices[0:train_split_idx]
    val_indices = shuffled_indices[train_split_idx:val_split_idx]
    test_indices = shuffled_indices[val_split_idx:]

    # Select the examples from x and y to construct our training, validation, testing sets
    x_train, y_train = x[train_indices, :], y[train_indices]
    x_val, y_val = x[val_indices, :], y[val_indices]
    x_test, y_test = x[test_indices, :], y[test_indices]

    '''
    Trains and tests logistic regression model from scikit-learn
    '''
    model_scikit = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000)

    # TODO: Train scikit-learn logistic regression model
    model_scikit.fit(x_train, y_train)

    print(f'\n***** Results on the ToyChess dataset using scikit-learn logistic regression model *****')

    # TODO: Score model using mean accuracy on training set
    predictions_train = model_scikit.predict(x_train)
    score_train = skmetrics.accuracy_score(y_train, predictions_train, normalize=True)
    print('Training set mean accuracy: {:.4f}'.format(score_train))

    # TODO: Score model using mean accuracy on validation set
    predictions_val = model_scikit.predict(x_val)
    score_val = skmetrics.accuracy_score(y_val, predictions_val, normalize=True)
    print('Validation set mean accuracy: {:.4f}'.format(score_val))

    # TODO: Score model using mean accuracy on testing set
    predictions_test = model_scikit.predict(x_test)
    score_test = skmetrics.accuracy_score(y_test, predictions_test, normalize=True)
    print('Testing set mean accuracy: {:.4f}\n'.format(score_test))


# Turn Nx8x8 board into Nx64 through PCA
def plot(x, y, title, xlabel, ylabel):
    '''
    Plots x against y with plot title

    Arg(s):
        x : numpy[float32]
            array of values
        y : numpy[float32]
            array of values
        title : str
            (super)title of plot
        xlabel : str
            label of x axis
        ylabel : str
            label of y axis
    '''

    # Create a 1 x 1 figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO: Plot y against x, set marker to 'o', color to 'b'
    ax.plot(x, y, marker='o', color='b')

    # TODO: Create super title for figure
    fig.suptitle(title)

    # TODO: Set x and y axis with labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

def plot_images(X, n_row, n_col, title, cmap='gray'):
    '''
    Creates n_row by n_col panel of images

    Args:
        X : numpy[float32]
            N x h x w numpy array
        n_row : int
            number of rows in figure
        n_col : list[str]
            number of columns in figure
        title : str
            title of plot
        cmap : str
            colormap to use for visualizing images
    '''

    fig = plt.figure()

    # TODO: Set title as figure super title
    fig.suptitle(title)

    # TODO: Iterate through X and plot the first n_row x n_col elements as figures
    # visualize using the specified 'gray' colormap
    # use plt.box(False) and plt.axis('off') to turn off borders and axis
    for i in range(0, n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(X[i], cmap=cmap)
        plt.axis('off')
        plt.box(False)

    plt.show()

class PrincipalComponentAnalysis(object):

    def __init__(self, d):
        # Number of eigenvectors to keep
        self.__d = d

        # Mean of the dataset
        self.__mean = None

        # Linear weights or transformation to project to lower subspace
        self.__weights = None

        # Eigenvalues of the dataset
        self.__eigenvalues = None

    def __center(self, X):
        '''
        Centers the data to zero-mean

        Args:
            X : numpy[float32]
                N x D feature vectors
        Returns:
            numpy[float32] : N x D centered feature vectors
        '''

        # TODO: Center the data

        return X - self.__mean

    def __covariance_matrix(self, X):
        '''
        Computes the covariance matrix of feature vectors

        Args:
            X : numpy[float32]
                N x D feature vectors
        Returns:
            numpy[float32] : D x D covariance matrix
        '''

        # TODO: Compute the covariance matrix
        N = X.shape[0]

        C = (1 / (N - 1)) * np.dot(X.T, X)

        return C

    def fit(self, X):
        '''
        Obtains the top d eigenvectors (weights) from the input feature vectors

        Arg(s):
            X : numpy[float32]
                N x D feature vector
        '''

        # TODO: Implement the fit function

        # Make sure that d is less or equal D
        assert self.__d <= X.shape[1]

        # TODO: Compute mean
        self.__mean = np.mean(X, axis=0)

        # Center the data
        X_centered = self.__center(X)

        # TODO: Compute the covariance matrix
        cov_matrix = self.__covariance_matrix(X_centered)

        # TODO: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # TODO: Store the top d eigenvalues
        idx = np.argsort(eigenvalues)[::-1]

        sorted_eigenvalues = eigenvalues[idx]

        sorted_eigenvectors = eigenvectors[:, idx]

        self.__eigenvalues = sorted_eigenvalues[:self.__d]

        # TODO: Select the top d eigenvectors
        self.__weights = sorted_eigenvectors[:, :self.__d]


    def project_to_subspace(self, X):
        '''
        Project data X to lower dimension subspace using the top d eigenvectors

        Arg(s):
            X : numpy[float32]
                N x D feature vectors
        Returns:
            numpy[float32] : N x d feature vectors
        '''
        # Center the data
        X_centered = self.__center(X)

        # TODO: Computes transformation to lower dimension and project to subspace

        return np.dot(X_centered, self.__weights)

    def get_eigenvalues(self):
        '''
        Returns eigenvalues

        Returns:
            numpy[float32] : eigenvalues
        '''

        return self.__eigenvalues

    def reconstruct_from_subspace(self, Z):
        '''
        Reconstruct the original feature vectors from the latent vectors

        Arg(s):
            Z : numpy[float32]
                N x d latent vectors
        Returns:
            numpy[float32] : N x D feature vectors
        '''

        # TODO: Reconstruct the original feature vector

        return np.dot(Z, self.__weights.T) + self.__mean

    def generate_new_samples(self, m):
        '''
        Generates new data points by sampling from

        Arg(s):
            m : int
                number of samples to generate
        Returns:
            numpy[float32] : m x D samples
        '''

        # TODO: Generate new samples
        # Generate random vectors
        new_X = np.random.normal(size=(m, self.__d))
        new_X *= np.sqrt(self.__eigenvalues[:self.__d])

        return self.reconstruct_from_subspace(new_X)

def second_version():
    # Load the dataset
    df = pd.read_csv('toyChessData.csv')  # Replace 'your_dataset.csv' with your actual file name

    # white is capital and positive, black is lowercase and negative
    key = {'r': 0, 'n': 1, 'b': 2, 'q': 3,'k':4, 'p': 5,'R': 6, 'N': 7, 'B': 8, 'Q': 9,'K': 10, 'P': 11}

    x = []
    y = []

    # Transforming
    for idx, row in df.head(1).iterrows():
        board = chess.Board(row['FEN'])

        bitboard = np.zeros((12, 64), dtype=np.int8)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                bitboard[key[piece.symbol()]][square] = 1
        # print(bitboard)
        # features = [np.packbits(bitboard[i]).view(np.uint64)[0] for i in range(12)]
        x.append(bitboard)

        eval = row['Evaluation'].strip('+')
        if '#' in eval:
            y.append(np.sign(int(eval[1:])) * 1000)
        else:
            y.append(int(eval))
    
    X = np.array(x[0])
    y = np.array(y)

    y[y > 0] = 1 # white wins, pos
    y[y <= 0] = 0 # black wins, neg

    # TODO: Get the number of dimensions in the dataset
    n_dim = X.shape[1]

    # TODO: Reshape handwritten digits dataset to N x 8 x 8
    X_reshaped = np.reshape(X, (X.shape[0], 8, 8))

    # TODO: Plot 3 x 3 panel of handwritten digits with title 'Handwritten digits dataset'
    plot_images(X_reshaped, 4, 3, title='Chess board dataset')

    # TODO: Vectorize handwritten digits dataset to N x D
    X_vectorized = np.reshape(X_reshaped, (X_reshaped.shape[0], 1, -1))

    # TODO: Plot 9 x 1 panel of handwritten digits with title 'Vectorized handwritten digits dataset'
    plot_images(X_vectorized, 12, 1, title='Vectorized chess board dataset')

second_version()
