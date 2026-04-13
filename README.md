# ⚡ AI Energy Forecasting using LSTM

An end-to-end Machine Learning project that predicts energy consumption using LSTM (Long Short-Term Memory) neural networks.

---

## 🚀 Features

* 📊 Time-series forecasting using LSTM
* 🔄 Data preprocessing & scaling
* 📈 Visualization of actual vs predicted values
* 🌐 REST API using Flask
* 🧪 Tested using Postman

---

## 📂 Project Structure

```
AI-Energy-Forecasting-LSTM/
│── data/
│   └── energy.csv
│── images/
│   └── output.png
│── models/
│   ├── lstm_model.h5
│   └── scaler.pkl
│── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│── app.py
│── main.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Energy-Forecasting-LSTM.git
cd AI-Energy-Forecasting-LSTM
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 1. Train Model

```bash
python main.py
```

### 2. Start API

```bash
python app.py
```

---

## 📡 API Endpoint

**POST** `/predict`

### Example Input:

```json
{
  "sequence": [100,120,130,150,160,170,180,200,220,250]
}
```

### Example Output:

```json
{
  "predicted_energy": 183.85
}
```

---

## 📊 Output Visualization

![Output](images/output.png<img width="960" height="540" alt="ss7" src="https://github.com/user-attachments/assets/df62df53-6fa1-4c69-9b9d-01b4286ab987" />
<img width="960" height="540" alt="ss6" src="https://github.com/user-attachments/assets/fe6f3dca-6839-4a29-8f67-5f5d521d7a71" />
<img width="960" height="540" alt="ss5" src="https://github.com/user-attachments/assets/dac83252-36de-4c1a-8f58-e5a6b2517b64" />
)

---

## ⚠️ Note

Model files may not be included due to size.
Run `main.py` to generate:

* `lstm_model.h5`
* `scaler.pkl`

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Pandas, NumPy
* Flask
* Matplotlib

---

## 👩‍💻 Author

Arpita Bhendigeri
