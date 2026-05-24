### 💐 Perfume Recommendation AI

An AI-powered web application built with **Streamlit** and **Scikit-Learn** that recommends perfumes based on user-specified fragrance profiles (accords) and target gender. The project uses a **TF-IDF Vectorizer** and **Cosine Similarity** to match natural language user queries with a dataset containing over 70,000 perfume records.

---

### 🚀 Live Demo
* can add your deployed app link here *

---

### ✨ Features
* **Intelligent Recommendations:** Matches custom input queries (e.g., *"citrus woody for women"*) with the dataset using advanced text vectorization.
* **Fast Search Response:** Pre-computed TF-IDF vectors are stored and cached using Streamlit's `@st.cache_resource` for instant results.
* **Clean & Modern UI:** An intuitive, interactive user interface built completely with Streamlit markdown and widgets.
* **Extensive Dataset:** Built on a database containing detailed records of thousands of perfumes, including names, main accords, and gender classifications.

---

## 📊 Dataset & Model Details
The underlying recommendation system processes data through the following pipeline:
1. **Feature Engineering:** Merges the `Main Accords` and `Gender` columns to build a comprehensive textual footprint for each perfume.
2. **Vectorization:** Converts textual features into a spatial matrix using a `TfidfVectorizer` (configured for optimal feature limits and English stop-word filtering).
3. **Similarity Analysis:** Utilizes `cosine_similarity` to mathematically calculate the closest matches between your input query and the 70,000+ entries.

---

### 📂 Project Structure
```text
├── AI_ASSISTANT_PERFUME.ipynb   # Jupyter Notebook for Data Cleaning & Model Training
├── app.py                       # Streamlit Web Application frontend & backend logic
├── fra_perfumes.csv             # Raw perfume dataset (70k+ entries)
├── tfidf.pkl                    # Serialized TF-IDF Vectorizer model
├── vectors.pkl                  # Serialized matrix of pre-computed feature vectors
├── data.pkl                     # Serialized Pandas DataFrame for UI rendering
└── requirements.txt             # List of python dependencies# AI-Perfume-Assistant

---

### 🛠️ Installation & Setup
Follow these steps to run this project locally on your machine:

1. Clone the Repository

Bash

git clone [https://github.com/AnzaKayani/Perfume-Recommendation-AI.git](https://github.com/AnzaKayani/Perfume-Recommendation-AI.git)
cd Perfume-Recommendation-AI

---

### Create Virtual Environment( Optional)

# Windows
python -m venv venv
venv\Scripts\activate

---

### Install Dependencies

Make sure you have a requirements.txt file setup with streamlit, pandas, and scikit-learn. Then run:

Bash

pip install -r requirements.txt

---

### Train the Model / Export Pickle Files
Run your training notebook or script (AI_ASSISTANT_PERFUME.ipynb) to process the raw dataset fra_perfumes.csv and generate the required pickle files (tfidf.pkl, vectors.pkl, and data.pkl).

---

### Launch the Web App

Bash

streamlit run app.py

---

### 💻 Tech Stack Used
Frontend UI: Streamlit
Data Manipulation: Pandas, NumP
Machine Learning Engine: Scikit-Learn (TfidfVectorizer, cosine_similarity)
Model Serialization: Pickle
