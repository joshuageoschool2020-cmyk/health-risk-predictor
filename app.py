import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import requests
from io import StringIO

st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="wide")

st.markdown("""<style>
.risk-box{padding:1.2rem 1.5rem;border-radius:12px;margin:1rem 0;text-align:center}
.risk-low{background:#e8f5e9;border:1.5px solid #43a047;color:#1b5e20}
.risk-mid{background:#fff8e1;border:1.5px solid #f9a825;color:#e65100}
.risk-high{background:#ffebee;border:1.5px solid #e53935;color:#b71c1c}
.ai-box{background:#f0f4ff;border-left:4px solid #3f51b5;border-radius:8px;padding:1rem 1.25rem;margin-top:1rem;font-size:0.95rem;line-height:1.8}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    RAW = """6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
1,103,30,38,83,43.3,0.183,33,0
1,115,70,30,96,34.6,0.529,32,1
3,126,88,41,235,39.3,0.704,27,0
8,99,84,0,0,35.4,0.388,50,0
7,196,90,0,0,39.8,0.451,41,1
9,119,80,35,0,29.0,0.263,29,1
11,143,94,33,146,36.6,0.254,51,1
10,125,70,26,115,31.1,0.205,41,1
7,147,76,0,0,39.4,0.257,43,1
1,97,66,15,140,23.2,0.487,22,0
13,145,82,19,110,22.2,0.245,57,0
5,117,92,0,0,34.1,0.337,38,0
5,109,75,26,0,36.0,0.546,60,0
3,158,76,36,245,31.6,0.851,28,1
3,88,58,11,54,24.8,0.267,22,0
6,92,92,0,0,19.9,0.188,28,0
10,122,78,31,0,27.6,0.512,45,0
4,103,60,33,192,24.0,0.966,33,0
11,138,76,0,0,33.2,0.420,35,0
9,102,76,37,0,32.9,0.665,46,1
2,90,68,42,0,38.2,0.503,27,1
4,111,72,47,207,37.1,1.390,56,1
3,180,64,25,70,34.0,0.271,26,0
7,133,84,0,0,40.2,0.696,37,0
7,106,92,18,0,22.7,0.235,48,0
9,171,110,24,240,45.4,0.721,54,1
7,159,64,0,0,27.4,0.294,40,0
0,180,66,39,0,42.0,1.893,25,1
1,146,56,0,0,29.7,0.564,29,0
2,71,70,27,0,28.0,0.586,22,0
7,103,66,32,0,39.1,0.344,31,1
7,105,0,0,0,0.0,0.305,24,0"""
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    df = pd.read_csv(StringIO(RAW), names=cols)
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[c] = df[c].replace(0, df[c].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000,random_state=42)
    model.fit(X_train_s,y_train)
    acc = accuracy_score(y_test,model.predict(X_test_s))
    return model,scaler,acc,cols[:-1]

def get_ai_explanation(patient,risk_pct,risk_level,api_key):
    prompt = f"Health assistant: Patient age {patient['Age']}, glucose {patient['Glucose']}, BMI {patient['BMI']}, BP {patient['BloodPressure']}. ML predicted {risk_pct:.0f}% diabetes risk ({risk_level}). Write 4 sentences: what risk means, 2 concerning factors, one lifestyle tip, reassuring close."
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json","x-api-key":api_key,"anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":300,"messages":[{"role":"user","content":prompt}]},
            timeout=20)
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"AI unavailable: {e}"

model,scaler,accuracy,feature_names = load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Anthropic API key",type="password",placeholder="sk-ant-...")
    st.divider()
    st.metric("Accuracy",f"{accuracy*100:.1f}%")
    st.metric("Algorithm","Logistic Regression")
    st.metric("Dataset","Pima Indians")

st.title("🩺 Health Risk Predictor")
st.markdown("Enter patient data and click **Predict** to get diabetes risk score")
st.divider()

c1,c2,c3 = st.columns(3)
with c1:
    age     = st.slider("Age (years)",18,80,35)
    glucose = st.slider("Glucose (mg/dL)",60,200,110)
    bmi     = st.slider("BMI",15.0,55.0,28.0,0.1)
with c2:
    bp      = st.slider("Blood Pressure (mmHg)",50,130,72)
    preg    = st.slider("Pregnancies",0,15,1)
    insulin = st.slider("Insulin (µU/mL)",0,300,80)
with c3:
    skin    = st.slider("Skin Thickness (mm)",0,60,29)
    dpf     = st.slider("Diabetes Pedigree",0.0,2.5,0.47,0.01)
    st.write("")
    predict_btn = st.button("🔍 Predict Risk",use_container_width=True,type="primary")

st.divider()

if predict_btn:
    patient = {"Pregnancies":preg,"Glucose":glucose,"BloodPressure":bp,
               "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
               "DiabetesPedigreeFunction":dpf,"Age":age}
    patient_df = pd.DataFrame([patient])
    scaled = scaler.transform(patient_df)
    prob   = model.predict_proba(scaled)[0][1]
    pct    = prob*100
    if pct<33:     level,css = "Low Risk 🟢","risk-low"
    elif pct<66:   level,css = "Moderate Risk 🟡","risk-mid"
    else:          level,css = "High Risk 🔴","risk-high"

    r1,r2,r3 = st.columns(3)
    with r1:
        st.subheader("Result")
        st.markdown(f'<div class="risk-box {css}"><h2>{pct:.1f}%</h2><b>{level}</b></div>',unsafe_allow_html=True)
    with r2:
        st.subheader("Vitals")
        st.metric("Glucose",f"{glucose} mg/dL",delta="Normal" if glucose<100 else "Elevated",delta_color="normal" if glucose<100 else "inverse")
        st.metric("BMI",f"{bmi:.1f}",delta="Healthy" if bmi<25 else "Overweight",delta_color="normal" if bmi<25 else "inverse")
        st.metric("Blood Pressure",f"{bp} mmHg",delta="Normal" if bp<80 else "Elevated",delta_color="normal" if bp<80 else "inverse")
    with r3:
        st.subheader("Feature Impact")
        fi = pd.Series(np.abs(model.coef_[0]),index=feature_names).sort_values()
        fig,ax = plt.subplots(figsize=(4,4))
        ax.barh(fi.index,fi.values,color=["#e53935" if w>fi.median() else "#90caf9" for w in fi.values])
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.patch.set_alpha(0)
        st.pyplot(fig,use_container_width=True)

    if api_key:
        with st.spinner("AI thinking..."):
            st.markdown(f'<div class="ai-box">{get_ai_explanation(patient,pct,level,api_key)}</div>',unsafe_allow_html=True)
    else:
        st.info("Add Anthropic API key in sidebar for AI explanation")

    with st.expander("📐 Linear algebra"):
        z = float(np.dot(scaled[0],model.coef_[0])+model.intercept_[0])
        st.code(f"Patient vector: {np.round(scaled[0],3).tolist()}")
        st.code(f"Weight vector:  {np.round(model.coef_[0],3).tolist()}")
        st.code(f"z = x·w + b  = {z:.4f}")
        st.code(f"sigmoid(z)   = {prob:.4f} = {pct:.1f}%")
    st.success("✅ Done!")
else:
    st.info("👈 Adjust sliders and click Predict Risk")

st.caption("⚠️ Educational only. Not a medical diagnosis.")
