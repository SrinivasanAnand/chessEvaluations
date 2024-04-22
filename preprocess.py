import pandas as pd
import numpy as np
import chess
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skmetrics 

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
    


