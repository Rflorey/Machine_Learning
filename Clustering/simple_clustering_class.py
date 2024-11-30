import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class SimpleClusterAnalysis:
    def __init__(self):
        """Initialize the clustering analysis class"""
        self.df = None
        self.clusters = None
        self.model = None
        self.inertia_scores = None
        self.scaler = StandardScaler() 
    
    def load_data(self, data_source, index_col=None, parse_dates=False):
        """Load data from CSV file or DataFrame"""
        # if isinstance(data_source, str):
        #     self.df = pd.read_csv(data_source)
        # else:
        #     self.df = data_source.copy()
        # return self
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source, 
                                index_col=index_col, 
                                parse_dates=parse_dates)
        else:
            self.df = data_source.copy()
        return self
    
    def create_dummies(self, columns, drop_first=True):
        """
        Create dummy variables and convert to numeric (0,1)
        
        Parameters:
        columns (list): List of columns to convert to dummies
        drop_first (bool): Whether to drop first dummy column to avoid multicollinearity
        """
        # Create dummy variables
        dummy_df = pd.get_dummies(self.df[columns], columns=columns, 
                                prefix=columns, drop_first=drop_first)
        
        # Convert to numeric (0,1)
        dummy_df = dummy_df.astype(int)
        
        # Drop original columns and join dummy columns
        self.df = self.df.drop(columns=columns)
        self.df = pd.concat([self.df, dummy_df], axis=1)
        
        return self
    
    def preprocess(self, 
                  drop_columns=None,
                  scale_columns=None):
        """Preprocess the data"""
        # Drop columns
        if drop_columns:
            self.df = self.df.drop(columns=drop_columns)
                
        # Scale numeric columns
        if scale_columns:
            for col, factor in scale_columns.items():
                self.df[col] = self.df[col] / factor
        
        return self
    
    def head(self):
        """Return the first 5 rows of the DataFrame"""
        return self.df.head()
    
    def find_elbow(self, max_k=10):
        """Calculate inertia for different k values"""
        k_values = list(range(1, max_k + 1))
        inertia = []
        
        # Calculate inertia for each k
        for k in k_values:
            model = KMeans(n_clusters=k, n_init='auto', random_state=0)
            model.fit(self.df)
            inertia.append(model.inertia_)
        
        # Store results
        self.inertia_scores = pd.DataFrame({
            'k': k_values,
            'inertia': inertia
        })
        
        return self
    
    def plot_elbow(self):
        """Plot the elbow curve"""
        if self.inertia_scores is None:
            raise ValueError("Run find_elbow() first")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.inertia_scores['k'], 
                self.inertia_scores['inertia'], 
                'bo-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Curve')
        plt.xticks(self.inertia_scores['k'])
        plt.grid(True)
        plt.show()
        
        return self
    def std_scaler(self, columns=None):
        """
        Standardize specified columns using StandardScaler
        
        Parameters:
        columns (list): List of column names to scale. If None, scales all columns.
        """
        if columns is None:
            # Scale all columns
            self.df = pd.DataFrame(
                self.scaler.fit_transform(self.df),
                columns=self.df.columns,
                index=self.df.index
            )
        else:
            # Scale only specified columns
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
        
        return self
    

    def fit(self, n_clusters=4):
        """Fit the clustering model"""
 
        # Fit KMeans
        self.model = KMeans(n_clusters=n_clusters, n_init='auto')
        self.clusters = self.model.fit_predict(self.df)
        
        # Add clusters to dataframe
        self.df[f'Cluster'] = self.clusters
        
        return self
    
    def plot(self, x_column, y_column):
        """Create a scatter plot of the clusters"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.df[x_column],
            self.df[y_column],
            c=self.clusters,
            cmap='winter'
        )
        plt.colorbar(scatter)
        plt.title(f'Clusters (k={len(np.unique(self.clusters))})')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
        
        return self
    
    def get_results(self):
        """Return the results DataFrame"""
        return self.df

# Examples:
# Example usage 1
# ''' Perform clustering analysis on shopping data '''
# clustering = SimpleClusterAnalysis()
# clustering.load_data("https://static.bc-edx.com/mbc/ai/m2/datasets/shopping-data.csv")
# clustering.create_dummies(['Card Type'])
# print(clustering.head())
# clustering.preprocess(
#     drop_columns=["CustomerID"],
#     scale_columns={"Annual Income": 1000}
# )
# print(clustering.head())
# clustering.fit(n_clusters=4)
# clustering.plot("Annual Income", "Spending Score")
# results = clustering.get_results()
# print(results.head())

# Example usage 2
''' Find the best k value for clustering '''
clustering = SimpleClusterAnalysis()
df_clustering = clustering.load_data("https://static.bc-edx.com/mbc/ai/m2/datasets/option-trades.csv", 
                         index_col="date", 
                         parse_dates=True
                        )
clustering.find_elbow(max_k=10)
clustering.plot_elbow()

