# Import necessary libraries
import pandas as pd              # For data manipulation and analysis
import arff                      # For loading ARFF files
from sklearn.preprocessing import LabelEncoder  # For converting categorical data to numeric
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.tree import DecisionTreeClassifier  # For building a Decision Tree model
from sklearn.metrics import accuracy_score       # For evaluating model performance

# Step 1: Load the ARFF file
# Open the ARFF file (replace "connect-4.arff" with your actual file path if different)
# Use the `arff.load()` function to load the file content as a dictionary
with open("connect-4.arff") as f:
    data = arff.load(f)

# Step 2: Convert ARFF data to a DataFrame
# `data['data']` contains the actual game data (board positions and outcome)
# `data['attributes']` contains the column names and types in the format [(name, type), ...]
# We extract only the column names for creating the DataFrame
df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

# Step 3: Preprocess the data
# For machine learning, categorical values need to be converted to numeric form.
# In Connect-4, the board positions are often stored as categorical values like 'x' (player 1),
# 'o' (player 2), and 'b' (blank). We need to encode these values as integers.

label_encoder = LabelEncoder()  # Initialize a label encoder

# Encode each board position column (all columns except the last one, which is the target label)
# This loop iterates through each column except the target column and encodes it
for column in df.columns[:-1]:  # We use df.columns[:-1] to exclude the target column (outcome)
    df[column] = label_encoder.fit_transform(df[column])

# Encode the target column as well
# This will convert values like 'win', 'loss', and 'draw' to integers
# df.columns[-1] is the last column, which we assume is the target
df[df.columns[-1]] = label_encoder.fit_transform(df[df.columns[-1]])

# Step 4: Separate the dataset into features (X) and target (y)
# `X` is the feature matrix containing all board positions (all columns except the target)
# `y` is the target vector (the last column) representing game outcomes (win, loss, draw)
X = df.iloc[:, :-1]  # Select all rows and all columns except the last one
y = df.iloc[:, -1]   # Select all rows for the last column only (target)

# Step 5: Split the data into training and test sets
# We use 70% of the data for training and 30% for testing.
# This helps us evaluate how well the model performs on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train a Decision Tree Classifier
# We initialize a Decision Tree Classifier and train it using our training data (X_train, y_train)
# Decision Trees are useful for classification tasks and can handle categorical data well
model = DecisionTreeClassifier()  # Initialize the classifier
model.fit(X_train, y_train)       # Train the model on the training data

# Step 7: Make predictions on the test set
# Use the trained model to predict the outcomes of the test data (X_test)
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
# Calculate the accuracy of the model by comparing the predicted labels (y_pred) to the true labels (y_test)
# `accuracy_score` returns the proportion of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
