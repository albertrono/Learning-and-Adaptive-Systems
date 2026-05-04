import pandas as pd
import numpy as np

# -------------------------------
# Define the Play Tennis dataset
# -------------------------------
data = {
    'Outlook':     ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                    'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                    'Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':    ['High','High','High','High','Normal','Normal',
                    'Normal','High','Normal','Normal','Normal','High',
                    'Normal','High'],
    'Wind':        ['Weak','Strong','Weak','Weak','Weak','Strong',
                    'Strong','Weak','Weak','Weak','Strong','Strong',
                    'Weak','Strong'],
    'PlayTennis':  ['No','No','Yes','Yes','Yes','No','Yes','No',
                    'Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)

# -------------------------------
# Function to compute entropy
# -------------------------------
def entropy(target_col):
    """Calculate the entropy of a target column (list or pandas Series)."""
    # Get value counts and probabilities
    counts = target_col.value_counts()
    probs = counts / counts.sum()
    # Entropy = -sum(p * log2(p))
    ent = -np.sum(probs * np.log2(probs))
    return ent

# -------------------------------
# Function to compute information gain
# -------------------------------
def information_gain(data, split_feature, target):
    """
    Compute Information Gain of splitting on 'split_feature'
    for the given 'target' column in the DataFrame.
    """
    # Total entropy before splitting
    total_entropy = entropy(data[target])
    
    # Weighted entropy after split
    weighted_entropy = 0
    total_instances = len(data)
    
    for value in data[split_feature].unique():
        subset = data[data[split_feature] == value]
        weight = len(subset) / total_instances
        weighted_entropy += weight * entropy(subset[target])
    
    # Information Gain
    ig = total_entropy - weighted_entropy
    return ig

# -------------------------------
# Calculate and display results
# -------------------------------
target_variable = 'PlayTennis'
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

print(f"Entropy of the whole dataset: {entropy(df[target_variable]):.4f}\n")

ig_values = {}
for feature in features:
    ig = information_gain(df, feature, target_variable)
    ig_values[feature] = ig
    print(f"Information Gain ({feature}): {ig:.4f}")

# Identify the best feature to split
best_feature = max(ig_values, key=ig_values.get)
print(f"\nMost feasible feature to split: {best_feature}")
