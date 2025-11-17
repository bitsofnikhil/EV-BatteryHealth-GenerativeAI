# 🚗⚡ EV Battery Health Prediction using Generative AI (VAE + Random Forest)

A complete Generative AI–powered system that predicts **EV Battery SOH (State of Health)** and **Remaining Useful Life (RUL)** using:

- A **Variational Autoencoder (VAE)** to generate high-quality synthetic EV battery data  
- A **Random Forest Regressor** trained on real + synthetic data  
- A simple, beginner-friendly **Gradio Chatbot** that predicts battery SOH from user inputs  

This project was developed as part of the **AI–ML Generative AI Internship by Edunet Foundation**.

---

## ⭐ Project Highlights

### 🔋 **1. Limited Real Data → Synthetic Data Generation**
- Real EV charging dataset was small and sparse  
- A **Variational Autoencoder (VAE)** was used to learn hidden battery patterns  
- The trained VAE generated **synthetic EV battery samples**
- This improved model generalization and prediction accuracy

### 📈 **2. Random Forest Battery Health Predictor**
- A Random Forest Regressor was trained on combined:
  - Real Battery Data  
  - VAE-generated Synthetic Data  
- Predicts:
  - **Degradation Rate (%)**
  - **State of Health (SOH)**
  - **Remaining Useful Life (RUL)**

### 🤖 **3. Interactive Chatbot (Gradio UI)**
The app allows users to input:
- Battery Temperature (°C)  
- Average Voltage (V)  
- Capacity (% rated)  
- Charge Cycles  
- EV Model  

Output:
- 🔋 SOH Prediction  
- 🔧 RUL Estimation  
- 🛠 Suggestions for improving battery life  

---

## 🧠 **Architecture Overview**

