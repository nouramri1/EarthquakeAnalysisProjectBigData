import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np

# Load  the data
df = pd.read_csv('Worldwide-Earthquake-database.csv')
df['datetime'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']].fillna(0).astype(int))

#  significant recent earthquakes
df = df[(df['YEAR'] >= 2010) & ((df['EQ_PRIMARY'] >= 7.5) | (df['MMI_INTENSITY'] == 'X') | (df['TSUNAMI'] == 1) | (df['DEATHS'] > 0) | (df['DAMAGE_$MIL'] >= 1.0))]

# Normalize the 'EQ_PRIMARY' magnitude for better visualization
scaler = MinMaxScaler(feature_range=(1, 10))
df['scaled_magnitude'] = scaler.fit_transform(df[['EQ_PRIMARY']])
df['normalized_depth'] = StandardScaler().fit_transform(df[['DEPTH']])
df['damage_millions'] = df['DAMAGE_$MIL'].fillna(0)  # Handle missing values for damage

# PCA for dimensionality reduction
pca = PCA(n_components=3)
pca_results = pca.fit_transform(df[['LATITUDE', 'LONGITUDE', 'normalized_depth']])
df['pca1'], df['pca2'], df['pca3'] = pca_results[:, 0], pca_results[:, 1], pca_results[:, 2]

# Clustering with KMeans
kmeans = KMeans(n_clusters=5)
df['cluster'] = kmeans.fit_predict(pca_results)

# t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df[['LATITUDE', 'LONGITUDE', 'normalized_depth']])
df['tsne1'], df['tsne2'] = tsne_results[:, 0], tsne_results[:, 1]

# 3D Scatter Plot using Plotly for PCA results
fig_pca = px.scatter_3d(df, x='pca1', y='pca2', z='pca3', color='cluster',
                        title='3D PCA of Earthquake Data', labels={'pca1': 'PC1', 'pca2': 'PC2', 'pca3': 'PC3'},
                        color_continuous_scale=px.colors.sequential.Viridis)
fig_pca.show()

# 3D Scatter Plot using Plotly for original geospatial data
fig_geo = px.scatter_3d(df, x='LONGITUDE', y='LATITUDE', z='DEPTH', color='scaled_magnitude',
                        title='3D View of Earthquake Intensity and Depth', labels={'DEPTH': 'Focal Depth (km)'},
                        color_continuous_scale=px.colors.sequential.Plasma)
fig_geo.show()

# 3D Visualization: Magnitude, Depth, and Economic Damage
fig_damage = px.scatter_3d(df, x='scaled_magnitude', y='normalized_depth', z='damage_millions', color='cluster',
                           title='3D Impact Analysis of Earthquakes', labels={'scaled_magnitude': 'Magnitude', 'normalized_depth': 'Depth', 'damage_millions': 'Damage in Millions USD'},
                           color_continuous_scale=px.colors.sequential.Cividis)
fig_damage.show()

# t-SNE Visualization
plt.figure(figsize=(10, 5))
sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', palette='coolwarm', data=df, legend='full', alpha=0.7)
plt.title('t-SNE Clustering of Earthquake Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)

# Plotting Daily Averages with Annotations
daily_avg = df.resample('D', on='datetime')['scaled_magnitude'].mean().dropna()
plt.figure(figsize=(14, 7))
plt.plot(daily_avg.index, daily_avg, label='Daily Average Magnitude', color='blue', marker='o')
plt.title('Daily Trends in Earthquake Magnitude')
plt.xlabel('Date')
plt.ylabel('Scaled Magnitude')
plt.axhline(y=daily_avg.mean(), color='red', linestyle='--', label='Overall Average')
plt.legend()
plt.grid(True)
plt.show()

