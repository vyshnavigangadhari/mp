

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
hostel_data = pd.read_csv("C:\\Users\\Vyshnavi G\\Desktop\\Data\\dataset1.csv")

# Print column names
print("Column Names in the Dataset:")
print(hostel_data.columns)

# 2. Preprocess the data (if needed)

# 3. User Input
try:
    Fee = float(input("Enter your maximum fee: "))
except ValueError:
    print("Please enter a valid number for the maximum fee.")
    exit()

Sharing = input("Enter your sharing preference (single/double): ").lower()  # Convert to lowercase

print("User input - Maximum Fee: {}, Sharing Preference: {}".format(Fee, Sharing))

# 4. Filter Dataset
# Convert 'Fee' column to numeric if it's not already numeric
hostel_data['Fee'] = pd.to_numeric(hostel_data['Fee'], errors='coerce')

# Convert 'Sharing' column to strings and lowercase
hostel_data['Sharing'] = hostel_data['Sharing'].astype(str).str.lower()

# Filter the dataset based on user input criteria
filtered_data = hostel_data[(hostel_data['Fee'] <= Fee) & (hostel_data['Sharing'] == Sharing)]

# 5. Similarity Calculation
# Assuming 'Fee' and 'Sharing' are relevant features for similarity calculation
def calculate_similarity(user_input, hostel_data):
    # Convert user input into DataFrame
    user_df = pd.DataFrame([user_input])

    # Calculate cosine similarity between user input and dataset
    similarity_scores = cosine_similarity(user_df, hostel_data[['Fee', 'Sharing']])
    return similarity_scores

# Calculate similarity scores
similarity_scores = calculate_similarity({'Fee': Fee, 'Sharing': Sharing}, filtered_data)

# 6. Recommendation
# Find top N hostels based on similarity scores
top_N = 5

if similarity_scores.max() < 0.5:  # Adjust the threshold as needed
    print("Sorry, no hostels match your preferences.")
else:
    top_hostels_indices = similarity_scores.argsort()[:, -top_N:][0][::-1]  # Sort in descending order
    recommended_hostels = filtered_data.iloc[top_hostels_indices]
    print("Top {} recommended hostels:".format(top_N))
    print(recommended_hostels)