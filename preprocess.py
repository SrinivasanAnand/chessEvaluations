import pandas as pd

# Load the dataset
df = pd.read_csv('chessData.csv')  # Replace 'your_dataset.csv' with your actual file name

# Select the first n rows (e.g., first 10 rows)
n = 100000
first_n_rows = df.head(n)

# Save the selected data to a new .csv file
first_n_rows.to_csv('toyChessData.csv', index=False)  # Replace 'first_n_rows.csv' with your desired file name
