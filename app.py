import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Hierarchical Clustering Tool", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_评估=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

linkage_method = st.sidebar.selectbox(
    "Linkage Method", 
    ["ward", "single", "complete", "average", "centroid"]
)

# --- Main App Logic ---
st.title("📊 Hierarchical Clustering Dashboard")
st.markdown("This app performs hierarchical clustering on your data and visualizes the relationship between points using a Dendrogram.")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Selection
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_cols = st.multiselect(
            "Select columns for clustering:", 
            options=numeric_cols, 
            default=numeric_cols
        )
        
    if len(selected_cols) >= 2:
        X = df[selected_cols]
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Linkage
        linked = linkage(X_scaled, method=linkage_method)
        
        with col2:
            st.subheader("2. Cluster Visualization")
            
            # Interactive Threshold
            max_d = st.slider("Distance Threshold (Cut-off Line)", 
                              min_value=0.0, 
                              max_value=float(np.max(linked[:, 2])), 
                              value=float(np.max(linked[:, 2]) * 0.7))
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(
                linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True,
                ax=ax
            )
            ax.axhline(y=max_d, color='r', linestyle='--')
            ax.set_title(f"Dendrogram (Method: {linkage_method})")
            ax.set_xlabel("Data Points")
            ax.set_ylabel("Euclidean Distance")
            
            st.pyplot(fig)

        # Cluster Assignment
        st.divider()
        st.subheader("3. Cluster Results")
        
        # Calculate cluster labels based on the threshold
        df['Cluster_Labels'] = fcluster(linked, max_d, criterion='distance')
        
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.write("Data with Cluster Labels:")
            st.dataframe(df, use_container_width=True)
        
        with res_col2:
            st.write("Cluster Distribution:")
            st.write(df['Cluster_Labels'].value_counts())
            
            # Download Result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='clustered_data.csv',
                mime='text/csv',
            )
    else:
        st.warning("Please select at least 2 numeric columns to perform clustering.")

else:
    # Landing Page State
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Hierarchical_clustering_dendrogram.png/800px-Hierarchical_clustering_dendrogram.png", width=400)
    st.markdown("""
    **How to use:**
    1. Upload a dataset (like your `D10_data.csv`).
    2. Select the numeric features you want to analyze.
    3. Adjust the **Linkage Method** to see how clusters form.
    4. Move the **Distance Threshold** slider to decide how many clusters to create.
    5. Download your final results.
    """)
