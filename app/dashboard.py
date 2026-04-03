import streamlit as st
import torch
import pandas as pd
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure Python can find your models folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_architecture import FraudGNN

# --- 1. PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="GNN Fraud Detection", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px;}
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 20px; font-style: italic;}
    .prediction-card { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;}
    .fraud { background-color: #fee2e2; border: 2px solid #ef4444; color: #b91c1c; }
    .normal { background-color: #dcfce3; border: 2px solid #22c55e; color: #15803d; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHED DATA LOADING ---
@st.cache_resource
def load_system():
    # Load graph data
    data = torch.load('data/graph_data.pt', weights_only=False)
    
    # Load model
    model = FraudGNN(num_node_features=30, num_classes=2)
    model.load_state_dict(torch.load('models/fraud_model.pth', weights_only=True))
    model.eval() 
    
    # Load raw dataset
    df = pd.read_csv('data/creditcard_processed.csv')
    
    # Pre-calculate overall accuracy and confusion matrix for the dataset tab
    with torch.no_grad():
        all_preds = model(data).argmax(dim=1).numpy()
    true_labels = data.y.numpy()
    cm = confusion_matrix(true_labels, all_preds)
    
    return data, model, df, cm

try:
    data, model, df, cm = load_system()
except Exception as e:
    st.error(f"System Error: Could not load components. Details: {e}")
    st.stop()

# --- 3. SESSION STATE FOR RANDOM BUTTON ---
if 'node_id' not in st.session_state:
    st.session_state.node_id = 150

def set_random_node():
    st.session_state.node_id = random.randint(0, data.num_nodes - 1)

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2916/2916362.png", width=80) 
    st.markdown("### ⚙️ Control Panel")
    
    # Testing Input
    st.session_state.node_id = st.slider(
        "Select Transaction ID:", 
        min_value=0, max_value=data.num_nodes-1, 
        value=st.session_state.node_id
    )
    
    st.button("🎲 Random Transaction", on_click=set_random_node, use_container_width=True)
    
    # st.markdown("---")
    # st.markdown("### 📚 Project Info")
    # st.write("**Model:** Graph Convolutional Network")
    # st.write("**Dataset:** Credit Card Fraud (Undersampled)")
    # st.write("**Submitted by:** [Your Name/Roll No]") # <-- CHANGE YOUR NAME HERE

# --- 5. MAIN HEADER ---
st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Graph Neural Networks (GNN) to detect anomalies in financial networks.</p>', unsafe_allow_html=True)

# --- 6. TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 Dataset Insights", "🧠 Model Architecture"])

# ==========================================
# TAB 1: LIVE PREDICTION
# --- LIVE PREDICTION TAB FIXED ---

with tab1:
    st.caption("This section analyzes a single transaction and its graph neighbors in real-time to predict fraud.")
    
    # Run Inference
    with torch.no_grad():
        output = model(data)
        probabilities = torch.exp(output[st.session_state.node_id])
        pred_class = output.argmax(dim=1)[st.session_state.node_id].item()
        
    actual_class = data.y[st.session_state.node_id].item()
    conf_score = probabilities[pred_class].item()
    
    col1, col2 = st.columns([1, 1.5])
    
    # ================= LEFT SIDE =================
    with col1:
        st.subheader("🤖 AI Prediction")

        # Big Visual Card
        if pred_class == 1:
            st.markdown(
                '<div class="prediction-card fraud"><h1>🚨 FRAUDULENT</h1><p>High risk of anomalous activity</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="prediction-card normal"><h1>✅ NORMAL</h1><p>Transaction appears legitimate</p></div>',
                unsafe_allow_html=True
            )

        # Confidence Score
        st.write(f"**AI Confidence:** {conf_score * 100:.1f}%")
        st.progress(conf_score)

        # Explanation (cleaned)
        with st.expander("💡 Why did the AI predict this?"):
            if pred_class == 1:
                st.write("""
                - This transaction shows abnormal feature values  
                - It is connected to suspicious neighboring transactions  
                - Pattern matches known fraud behavior  
                """)
            else:
                st.write("""
                - Transaction features are within normal range  
                - Neighboring transactions are also normal  
                - Pattern matches legitimate activity  
                """)

        # Ground Truth
        if pred_class == actual_class:
            st.success(f"**Model is Correct!** Actual record was also Class {actual_class}.")
        else:
            st.error(f"**Model is Incorrect.** Actual record was Class {actual_class}.")

    # ================= RIGHT SIDE =================
    with col2:
        st.subheader("📋 Transaction Signature")
        
        node_features = df.iloc[st.session_state.node_id].drop(['Class', 'Amount']).values[:15]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(node_features)), node_features, color='#3b82f6')
        ax.set_title(f"Feature Pattern for Transaction #{st.session_state.node_id}")
        ax.set_xlabel("PCA Features (V1 - V15)")
        ax.set_ylabel("Value")
        ax.axhline(0, color='black', linewidth=0.5)
        st.pyplot(fig)

        st.caption("Higher spikes indicate abnormal behavior compared to normal transactions.")
        st.write(f"**Transaction Amount:** ${df.iloc[st.session_state.node_id]['Amount']:.2f}")
    # Run Inference
    with torch.no_grad():
        output = model(data)
        probabilities = torch.exp(output[st.session_state.node_id])
        pred_class = output.argmax(dim=1)[st.session_state.node_id].item()
        
    actual_class = data.y[st.session_state.node_id].item()
    conf_score = probabilities[pred_class].item()
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🤖 AI Prediction")
        # ===============================
# Reason for Prediction (NEW)
# ===============================
        # Big Visual Card
    if pred_class == 1:
            st.markdown('<div class="prediction-card fraud"><h1>🚨 FRAUDULENT</h1><p>High risk of anomalous activity</p></div>', unsafe_allow_html=True)
    else:
            st.markdown('<div class="prediction-card normal"><h1>✅ NORMAL</h1><p>Transaction appears legitimate</p></div>', unsafe_allow_html=True)
            
        # Confidence Score Progress Bar
    st.write(f"**AI Confidence:** {conf_score * 100:.1f}%")
    st.progress(conf_score)
        
        # Simple Explanation for Viva
    with st.expander("💡 Why did the AI predict this?"):
            if pred_class == 1:
                st.write("The model predicted this is **Fraud** because its features and its connected neighbors in the graph closely match known fraudulent patterns.")
            else:
                st.write("The model predicted this is **Normal** because its transaction behavior is consistent with legitimate nodes in the network.")
        
        # Ground Truth comparison
    if pred_class == actual_class:
            st.success(f"**Model is Correct!** Actual record was also Class {actual_class}.")
    else:
            st.error(f"**Model is Incorrect.** Actual record was Class {actual_class}.")

    # with col2:
    #     st.subheader("📋 Transaction Signature")
        
    #     # Extract features for this node (first 15 PCA features for a clean chart)
    #     node_features = df.iloc[st.session_state.node_id].drop(['Class', 'Amount']).values[:15] 
        
    #     # Plot transaction signature
    #     fig, ax = plt.subplots(figsize=(8, 3))
    #     ax.bar(range(len(node_features)), node_features, color='#3b82f6')
    #     ax.set_title(f"Feature Pattern for Transaction #{st.session_state.node_id}")
    #     ax.set_xlabel("PCA Features (V1 - V15)")
    #     ax.set_ylabel("Value")
    #     ax.axhline(0, color='black', linewidth=0.5)
    #     st.pyplot(fig)
        
    #     st.write(f"**Transaction Amount:** ${df.iloc[st.session_state.node_id]['Amount']:.2f}")

# ==========================================
# TAB 2: DATASET INSIGHTS
# ==========================================
with tab2:
    st.caption("Explore the balanced dataset and overall model performance across all graph nodes.")
    
    # Metrics Container
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Nodes", data.num_nodes)
        m2.metric("Network Edges", data.num_edges)
        m3.metric("Normal Class", len(df[df['Class']==0]))
        m4.metric("Fraud Class", len(df[df['Class']==1]))
        
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Class Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        ax_pie.pie([len(df[df['Class']==0]), len(df[df['Class']==1])], 
                   labels=['Normal', 'Fraud'], colors=['#22c55e', '#ef4444'], 
                   autopct='%1.1f%%', startangle=90)
        st.pyplot(fig_pie)
        
    with c2:
        st.subheader("Model Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred Normal', 'Pred Fraud'], 
                    yticklabels=['True Normal', 'True Fraud'], ax=ax_cm)
        st.pyplot(fig_cm)

# ==========================================
# TAB 3: MODEL ARCHITECTURE
# ==========================================
with tab3:
    st.caption("How Graph Convolutional Networks (GCN) process transaction data.")
    
    st.markdown("### Architecture Flowchart")
    
    # Visual Flowchart using columns
    flow1, arrow1, flow2, arrow2, flow3, arrow3, flow4 = st.columns([2, 1, 2, 1, 2, 1, 2])
    
    with flow1:
        with st.container(border=True):
            st.markdown("#### 🔢 1. Input")
            st.write("30 Node Features")
            st.write("(PCA & Amount)")
            
    with arrow1:
        st.markdown("<h1 style='text-align: center; margin-top: 20px;'>➡️</h1>", unsafe_allow_html=True)
        
    with flow2:
        with st.container(border=True):
            st.markdown("#### 🕸️ 2. Graph")
            st.write("K-Nearest Neighbors")
            st.write("(Connect similar nodes)")
            
    with arrow2:
        st.markdown("<h1 style='text-align: center; margin-top: 20px;'>➡️</h1>", unsafe_allow_html=True)
        
    with flow3:
        with st.container(border=True):
            st.markdown("#### 🧠 3. GCN Layers")
            st.write("Conv1 -> ReLU")
            st.write("Conv2 -> Hidden patterns")
            
    with arrow3:
        st.markdown("<h1 style='text-align: center; margin-top: 20px;'>➡️</h1>", unsafe_allow_html=True)
        
    with flow4:
        with st.container(border=True):
            st.markdown("#### 🎯 4. Output")
            st.write("Log Softmax")
            st.write("0 (Normal) or 1 (Fraud)")

    st.markdown("---")
    st.markdown("### Why Graph Neural Networks?")
    st.info("""
    Traditional Machine Learning (like Random Forest) looks at transactions as **isolated events**. 
    Our GNN treats the dataset as a **Network**. It analyzes not just the transaction itself, but also its relationship 
    with neighboring transactions. Fraudsters often work in rings or show similar behavioral signatures—the GNN captures these hidden structural relationships, leading to more robust detection!
    """)