# 🚗 Week 2 – Generative AI for Electric Vehicle Battery Health

This week focuses on building a **Variational Autoencoder (VAE)** model using **Generative AI** to generate **synthetic battery data** and analyze **battery degradation** patterns in Electric Vehicles (EVs).

---

## 🎯 Objectives

- Understand how **Generative AI** can enhance **EV battery health analysis**.  
- Build and train a **Variational Autoencoder (VAE)** to learn battery performance features.  
- Generate **synthetic battery charging data** similar to real-world datasets.  
- Visualize the **latent space** and **training loss** of the model.

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| ML Framework | TensorFlow / Keras |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab |
| Dataset Source | Kaggle (EV Battery Charging Data) |

---

## 📊 Dataset Overview

Dataset: `ev_battery_charging_data.csv`  
**Shape:** (1000, 13)  
**Key Columns:**
- `SOC (%)` — State of Charge  
- `Voltage (V)`  
- `Current (A)`  
- `Battery Temp (°C)`  
- `Ambient Temp (°C)`  
- `Degradation Rate (%)`  
- `Efficiency (%)`  
- `Battery Type`, `Charging Mode`, `EV Model`

---

## ⚙️ Model Architecture

The **VAE (Variational Autoencoder)** consists of:

### 🔹 Encoder:
- Dense (64, relu)  
- Dense (32, relu)  
- Dense (`z_mean`, `z_log_var`)

### 🔹 Sampling Layer:
Generates latent variable `z = z_mean + exp(0.5 * z_log_var) * ε`

### 🔹 Decoder:
- Dense (32, relu)  
- Dense (64, relu)  
- Dense (output layer, sigmoid)

### 🔹 Loss:
Total Loss = Reconstruction Loss + KL Divergence

---

## 🧩 Training Details

| Parameter | Value |
|------------|--------|
| Epochs | 50 |
| Batch Size | 32 |
| Optimizer | Adam |
| Latent Dimension | 4 |

### ✅ Output during training:
Epoch 1/50
loss: 0.1218 - reconstruction_loss: 0.1110
...
Epoch 50/50
loss: 0.1081 - reconstruction_loss: 0.1081

yaml
Copy code

---

## 📈 Results

### 🪄 1. Sample Synthetic EV Battery Data Generated
| SOC (%) | Voltage (V) | Current (A) | Efficiency (%) | Battery Type | EV Model |
|----------|--------------|--------------|----------------|---------------|-----------|
| 56.09 | 3.86 | 54.09 | 97.97 | Li-ion | Model A |
| 54.16 | 3.85 | 55.14 | 97.97 | LiFePO4 | Model B |
| 54.78 | 3.85 | 54.77 | 97.99 | LiFePO4 | Model C |

---

### 📉 2. Training Loss Curve
![Training Loss Curve](../Week2/loss_curve.png)

---

### 🌌 3. Latent Space Visualization
Colored by **Degradation Rate (%)**

![Latent Space](../Week2/latent_space.png)

---

## 🔍 Insights

- The **VAE model** successfully learned latent battery features.
- **Synthetic samples** generated closely resemble real EV data.
- Latent space shows clustering based on **battery degradation**.
- This technique can augment small datasets for **EV battery prediction models**.

---

## 🧩 Next Steps (Week 3 Preview)

- Use the **synthetic data** for supervised training of a **Battery Health Prediction model**.
- Compare performance with and without generative data.
- Visualize model accuracy improvements.

---

## 👨‍💻 Author

**Nikhil Kumawat**  
B.Tech CSE, Gurukula Kangri Vishwavidyalaya  
[GitHub](https://github.com/bitsofnikhil) • [LinkedIn](https://www.linkedin.com/in/nikhil-kumawat-92a684291)

---

> 💡 *This project is part of the AICTE–Edunet Internship on “Generative AI for Electric Vehicles”.
