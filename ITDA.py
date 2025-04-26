import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Page Config ---
st.set_page_config(page_title="IoT Anomaly Detection", layout="wide")

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("## 🔐 IoT Anomaly Detection: Group G")
    st.image("https://images.icon-icons.com/2530/PNG/512/iot_button_icon_151911.png", width=100)
    st.markdown("Welcome to the IoT Traffic Anomaly Detection System. This tool leverages ML algorithms to help identify potential threats or abnormal behavior in your IoT network.")
    
    st.markdown("### 📊 Features")
    st.markdown("- Packet size analysis")
    st.markdown("- Port inspection")
    st.markdown("- Real-time anomaly detection")
    
    st.markdown("### 🧠 Models Included")
    st.markdown("- Random Forest (default)")
    st.markdown("- Gradient Boosting")
    st.markdown("- Logistic Regression")
    
    st.markdown("---")
    st.caption("🚀 Developed for anomaly detection in smart environments.")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\manqo\Downloads\archive (2)\iot_traffic_data.csv", header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df['Timestamp'] = df['Timestamp'].astype(str)

    numeric_cols = ['Src_Port', 'Dst_Port', 'Packet_Size', 'Payload', 'Label']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    df['Flags'].fillna('Unknown', inplace=True)
    df.dropna(subset=['Label'], inplace=True)
    df['Label'] = df['Label'].astype(int)
    return df

df = load_data()

# --- Preprocessing ---
def preprocess(df):
    X = df[['Src_Port', 'Dst_Port', 'Packet_Size', 'Payload']]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

X_train_scaled, X_test_scaled, y_train, y_test = preprocess(df)

# --- Train Models ---
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

trained_models = train_models(X_train_scaled, y_train)

# --- Evaluate Models ---
def evaluate(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

# --- Main Interface ---
st.title("📡 IoT Network Traffic Anomaly Detection")

model_choice = st.selectbox("🔍 Choose a model to evaluate:", ["Random Forest", "Gradient Boosting", "Logistic Regression"], index=0)
selected_model = trained_models[model_choice]
accuracy, report, cm = evaluate(selected_model, X_test_scaled, y_test)

# --- EDA ---
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Label Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Label', data=df, ax=ax1)
    ax1.set_title("Normal vs Anomalous Traffic")
    st.pyplot(fig1)
    st.markdown("✅ **Insight:** A higher number of normal traffic samples may indicate a well-functioning network. Anomalous traffic highlights potential security threats.")
    st.success("**Action:** Monitor spikes in anomalies to trigger automatic alerts.")

with col2:
    st.subheader("📦 Packet Size Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Packet_Size'], bins=50, kde=True, ax=ax2)
    ax2.set_title("Packet Size Histogram")
    st.pyplot(fig2)
    st.markdown("✅ **Insight:** Abnormally large or small packets may suggest scanning behavior or data exfiltration attempts.")
    st.success("**Action:** Flag packet sizes that fall outside historical norms for further inspection.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🔢 Source Port Activity")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, y='Src_Port', ax=ax3)
    ax3.set_title("Source Port Variability")
    st.pyplot(fig3)
    st.markdown("✅ **Insight:** Random high-numbered source ports are expected. Repeated use of specific ports could suggest malware communication.")
    st.success("**Action:** Set thresholds for unusual port usage frequency.")

with col4:
    st.subheader("🔢 Destination Port Usage")
    fig4, ax4 = plt.subplots()
    top_dst_ports = df['Dst_Port'].value_counts().head(10)
    sns.barplot(x=top_dst_ports.values, y=top_dst_ports.index, ax=ax4)
    ax4.set_title("Top 10 Destination Ports")
    st.pyplot(fig4)
    st.markdown("✅ **Insight:** Critical services using specific ports (e.g., 80, 443) are most targeted. Unexpected ports may imply backdoors.")
    st.success("**Action:** Audit traffic to uncommon destination ports.")

# --- Model Performance ---
st.header(f"📋 Model Performance: {model_choice}")
st.markdown(f"**✅ Accuracy:** `{accuracy:.4f}`")
st.markdown("**🧾 Classification Report**")
st.dataframe(pd.DataFrame(report).transpose())

# --- Confusion Matrix ---
st.subheader("📌 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)
st.markdown("✅ **Insight:** A high true positive rate means better detection of real threats. High false negatives are dangerous as attacks go unnoticed.")
st.success("**Action:** Continuously refine training data and use ensemble methods.")

# --- Feature Importance ---
if model_choice in ["Random Forest", "Gradient Boosting"]:
    st.subheader("⭐ Feature Importance")
    importances = selected_model.feature_importances_
    features = ['Src_Port', 'Dst_Port', 'Packet_Size', 'Payload']
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    st.dataframe(fi_df)

    fig_fi, ax_fi = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax_fi)
    ax_fi.set_title('Feature Importance')
    st.pyplot(fig_fi)
    st.markdown("✅ **Insight:** Knowing which features influence anomaly predictions helps prioritize data collection and optimization.")
    st.success("**Action:** Focus on improving the quality and consistency of top-ranked features.")

# --- Organizational Recommendations ---
st.header('📌 Organizational Insights & Recommendations')

st.info("**1. Real-time Anomaly Detection:** Enables rapid response to unusual device behavior.")
st.success("Benefit: Mitigates cyber threats like DDoS or data exfiltration early.")

st.info("**2. Improved Cybersecurity Posture:** Informs IT security teams of risks proactively.")
st.success("Benefit: Reduces breach risks and strengthens system defenses.")

st.info("**3. Data-Driven Optimization:** Supports smarter network resource management.")
st.success("Benefit: Helps organizations optimize bandwidth, device placement, and policies.")

st.info("**4. Compliance & Trust:** Aligns with security best practices.")
st.success("Benefit: Builds customer trust and eases regulatory compliance.")

# --- Footer ---
st.markdown("---")
st.caption("👨‍💻 Developed for Smart IoT Security | Built by Group G")
