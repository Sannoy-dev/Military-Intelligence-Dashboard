# AI-Based Military Intelligence Dashboard

An interactive AI-powered military intelligence dashboard built with **Streamlit**, **Machine Learning**. The application enables users to explore global terrorism incidents, analyze trends, predict attack types, forecast future incidents, and generate intelligence reports through a modern analytical interface.

---

## Features

- Interactive Dashboard
- Global Threat Map
- Country-wise Intelligence Analysis
- Attack Type Prediction using Machine Learning
- Threat Level Prediction
- Terrorism Forecasting
- AI Intelligence Report Generator
- Advanced Data Explorer
- Dashboard Settings

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib

---

## Machine Learning Models

The dashboard includes:

- Random Forest Classifier for Attack Type Prediction
- Random Forest Classifier for Threat Level Prediction
- Linear Regression for Attack Forecasting

---

## Dataset

This project uses the column reference of **Global Terrorism Database (GTD)** ["https://www.kaggle.com/datasets/START-UMD/gtd?select=globalterrorismdb_0718dist.csv"].

The dataset is **not included** in this repository due to its size.
Dataset need to be uploaded and columns of uploaded dataset must be mapped with the internal preset columns before training the data.

Download it from the official GTD website and place it inside:

```
data/
```

Expected file:

```
data/globalterrorismdb_0718dist --> or user can choose any csv file
```

---

## Project Structure

```
Military_Intelligence_Dashboard/
│
├── .gitignore
├── README.md
├── Main.py
│
├── assets/
│   └── style.css
│
├── data/
│   └── custom_dataset.csv
│
├── models/
│   └── custom/
│       ├── attack_prediction_label_encoder.joblib
│       ├── attack_prediction_metadata.json
│       ├── attack_prediction_model.joblib
│       ├── threat_level_metadata.json
│       └── threat_level_model.joblib
│
├── pages/
│   ├── Attack_prediction.py
│   ├── Country_analysis.py
│   ├── Data_explorer.py
│   ├── Forecasting.py
│   ├── Global_threat_map.py
│   ├── Home.py
│   ├── Intelligence_report.py
│   ├── Settings.py
│   └── Threat_level_prediction.py
│
└── utils/
    ├── data_loader.py
    ├── data_mapper.py
    ├── model_trainer.py
    └── ui.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Military-Intelligence-Dashboard.git
```

Go to the project folder:

```bash
cd AI-Military-Intelligence-Dashboard
```

Create a virtual environment(If needed):

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Required Files

Before running the project, place the following files in the appropriate folders:

### Dataset

```
data/[user's .csv file]
```

### Trained Models

```
models/
    ├── attack_prediction_label_encoder.joblib
    ├── attack_prediction_metadata.json
    ├── attack_prediction_model.joblib
    ├── threat_level_metadata.json
    └── threat_level_model.joblib
```

---

## Dashboard Modules

- Home Dashboard
- Global Threat Map
- Country Analysis
- Attack Prediction
- Threat Level Prediction
- Forecasting
- Intelligence Report
- Data Explorer
- Settings

---

## Future Improvements

- Deep Learning Models
- LSTM Time-Series Forecasting
- Real-time Intelligence Feeds
- GIS Heat Maps
- User Authentication
- Report Export (PDF)
- Interactive Alert System

---

## License

This project is developed for educational and research purposes.

The Global Terrorism Database (GTD) is maintained by the National Consortium for the Study of Terrorism and Responses to Terrorism (START). Please follow their licensing and usage terms when using the dataset.

---

## Author

**Sannoy Jana**

Computer Science & Engineering Student

AI • Machine Learning • Data Analytics 
