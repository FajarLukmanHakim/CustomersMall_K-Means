import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns


# your other code...

fig, ax = plt.subplots(figsize=(10, 5))


st.set_page_config(page_title="K-Means Mall Customer Clustering", layout="wide")

st.title("Customers Mall Dengan Menggunakan Algoritma K-Means")
st.title("Tugas Akhir Praktikum")
st.write("Upload dataset atau gunakan dataset default untuk clustering pelanggan mall.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("CustomerMall.csv")

if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

df.columns = ['Gender', 'Age', 'Annual_Income', 'Spending_Score(1-100)']
st.subheader("Informasi Dataset")
st.write(df.head())
st.write(df.describe())

# Correlation Heatmap
st.subheader("Heatmap Korelasi")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='viridis', ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("Pairplot Dataset")
st.write("Klik di area kosong jika plot tidak muncul sepenuhnya.")
pairplot_fig = sns.pairplot(df, hue="Gender", palette="tab10")
st.pyplot(pairplot_fig.fig)

# Boxplot
st.subheader("Boxplot")
fig, ax = plt.subplots(figsize=(10, 5))
df[['Age', 'Annual_Income', 'Spending_Score(1-100)']].boxplot(ax=ax)
st.pyplot(fig)

# KDE plots
st.subheader("Distribusi (KDE)")
fig, ax = plt.subplots(figsize=(10, 5))
df['Age'].plot.kde(label='Age')
df['Annual_Income'].plot.kde(label='Annual Income')
df['Spending_Score(1-100)'].plot.kde(label='Spending Score')
plt.legend()
st.pyplot(fig)

# Barplot Gender vs Annual Income
st.subheader("Barplot: Gender vs Annual Income")
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x='Gender', y='Annual_Income', data=df, ax=ax)
st.pyplot(fig)

# Scatterplots
st.subheader("Scatterplots")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x='Age', y='Annual_Income', hue='Gender', data=df, s=100, palette='tab10', ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x='Spending_Score(1-100)', y='Annual_Income', hue='Gender', data=df, s=100, palette='tab10', ax=ax)
st.pyplot(fig)

# KMeans Clustering
st.subheader("Menentukan Jumlah Cluster (Elbow Method)")
cluster_range = range(1, 12)
wcss = []

features = df[['Age', 'Annual_Income', 'Spending_Score(1-100)']].copy()
features_encoded = features.copy()
# Encode Gender
features_encoded['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_encoded)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
plt.plot(cluster_range, wcss, marker='o')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS')
plt.title('Elbow Method untuk Menentukan Cluster Optimal')
st.pyplot(fig)

n_clusters = st.sidebar.slider("Pilih jumlah cluster untuk K-Means", 2, 10, 4)

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
df['Cluster_No'] = kmeans.fit_predict(features_encoded)

# 3D Scatter plot
st.subheader("Visualisasi 3D Cluster")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'grey']

for cluster_num in range(n_clusters):
    ax.scatter(
        df['Age'][df['Cluster_No'] == cluster_num],
        df['Annual_Income'][df['Cluster_No'] == cluster_num],
        df['Spending_Score(1-100)'][df['Cluster_No'] == cluster_num],
        s=100,
        color=colors[cluster_num % len(colors)],
        label=f'Cluster {cluster_num}'
    )

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
ax.view_init(10, 170)
plt.legend()
st.pyplot(fig)

# Display groupby tables
st.subheader("Rata-rata Setiap Cluster")
st.dataframe(df.groupby('Cluster_No')[['Age', 'Annual_Income', 'Spending_Score(1-100)']].mean())

st.subheader("Minimum Setiap Cluster")
st.dataframe(df.groupby('Cluster_No')[['Age', 'Annual_Income', 'Spending_Score(1-100)']].min())

st.subheader("Maksimum Setiap Cluster")
st.dataframe(df.groupby('Cluster_No')[['Age', 'Annual_Income', 'Spending_Score(1-100)']].max())

st.subheader("Standar Deviasi Setiap Cluster")
st.dataframe(df.groupby('Cluster_No')[['Age', 'Annual_Income', 'Spending_Score(1-100)']].std())

st.success("Selesai. Kamu dapat mengunduh script ini untuk langsung dijalankan di Colab atau lokal.")

