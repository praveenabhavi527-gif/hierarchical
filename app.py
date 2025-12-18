import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Hierarchical Clustering Pro", layout="wide")

# Custom CSS for design
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_index=True)

# Sidebar Configuration
st.sidebar.header("🛠️ Configuration")
linkage_method = st.sidebar.selectbox(
    "Linkage Method", 
    ['ward', 'single', 'complete', 'average', 'centroid'],
    index=0
)
scaling_option = st.sidebar.checkbox("Scale Data (StandardScaler)", value=True)
color_threshold = st.sidebar.slider("Color Threshold (Cut-off)", 0, 100, 50)

st.title("📊 Hierarchical Clustering Dashboard")
st.markdown("Upload your CSV file to perform hierarchical clustering and visualize the dendrogram.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🌳 Dendrogram", "🧪 Cluster Results"])
    
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Selection
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols)

    if len(selected_cols) >= 2:
        X = df[selected_cols]
        
        # Preprocessing
        if scaling_option:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Perform Clustering
        with st.spinner('Calculating linkage...'):
            linked = linkage(X_scaled, method=linkage_method)

        with tab2:
            st.subheader("Hierarchical Clustering Dendrogram")
            fig, ax = plt.subplots(figsize=(12, 7))
            
            dendrogram(
                linked,
                ax=ax,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True,
                color_threshold=color_threshold,
                leaf_font_size=10
            )
            
            plt.title(f"Dendrogram (Method: {linkage_method})", fontsize=15)
            plt.xlabel("Data Points Index", fontsize=12)
            plt.ylabel("Euclidean Distance", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
        with tab3:
            st.subheader("Extracted Clusters")
            num_clusters = st.number_input("Enter number of clusters to extract", min_value=1, max_value=20, value=3)
            
            # Form clusters
            df['Cluster_ID'] = fcluster(linked, num_clusters, criterion='maxclust')
            
            st.write(f"Data with assigned Cluster IDs (Total Clusters: {num_clusters})")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Data as CSV",
                data=csv,
                file_name='clustered_data.csv',
                mime='text/csv',
            )
    else:
        st.warning("Please select at least 2 numeric columns to perform clustering.")

else:
    st.info("👋 Please upload a CSV file in the sidebar to get started.")
    # Example format info
    with st.expander("View required CSV format"):
        st.write("Your CSV should look like this:")
        example_df = pd.DataFrame({
            'x': [6.42, 8.22, 6.09],
            'y': [3.41, 5.56, 4.22],
            'label': [1, 1, 1]
        })
        st.table(example_df)
