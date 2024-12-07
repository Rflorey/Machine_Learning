import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs


# Generate synthetic data
# Parameters:
# - n_samples: List specifying number of samples for each class [1000, 100]
# - centers: None (automatically determined)
# - random_state: 1 (for reproducibility)
# - cluster_std: Different standard deviations for each cluster [1.3, 1]
# - center_box: Range for cluster centers (-3, 3)
X, y = make_blobs(n_samples = [1000, 100], centers=None, random_state=1, cluster_std=[1.3, 1], center_box = (-3, 3))

# Create DataFrame with features X1, X2 and target y
df = pd.DataFrame(X, columns = ["X1", "X2"])
df['y'] = y

# Save to CSV file
df.to_csv('ML_Optimization/data/blobs_data.csv', index = False)


# # Visualizing both classes
# plt.scatter(X[:, 0], X[:, 1], c=y)

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Imbalanced Dataset Visualization')
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar(label='Class')
plt.grid(True, alpha=0.3)
plt.show()

# Print dataset summary
print("\nDataset Summary:")
print(f"Total samples: {len(df)}")
print("\nClass distribution:")
print(df['y'].value_counts())
