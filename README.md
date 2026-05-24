
### 💐 Perfume Recommendation AI

An AI-powered web application built with **Streamlit** and **Scikit-Learn** that provides instant perfume recommendations based on user-specified fragrance profiles (accords) and target gender. By leveraging natural language processing techniques—specifically a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** combined with **Cosine Similarity**—the system maps user inputs against a dataset of over 70,000 perfume records to find the closest olfactory matches.

---

## 🚀 Live Demo
 Streamlit Community Cloud, live link here:
👉 [https://ai-perfume-assistant-2h83ngsdyaefcgxbq6wueq.streamlit.app/]

---

## ✨ Core Features
* **Natural Language Querying:** Users can enter rich, descriptive queries (e.g., *"fresh citrus and white floral for women"*, *"warm spicy tobacco amber for men"*) to discover tailored fragrance profiles.
* **Textual Feature Fusion:** Combined the `Main Accords` and `Gender` data into a unified text feature space to ensure target audience and olfactory characteristics are searched simultaneously.
* **Pre-computed Vector Alignment:** Text arrays are vectorized using an optimized `TfidfVectorizer` (capped at 3,000 features with English stop-words filtered out) to create low-latency search matrices.
* **High-Performance Memory Caching:** Implements Streamlit's `@st.cache_resource` decorator to load pre-trained vectorizers and matrix data-frames instantly into memory, reducing runtime compute overhead.
* **Sleek Interface Layout:** Designed a clean, minimalist UI using custom Streamlit markdown components, emojis, and status widgets for fluid interactive feedback.

---

### 📊 Technical Architecture & System Pipeline

The machine learning recommendation engine runs across a two-stage operational lifecycle:

1. **Training and Vectorization Phase (`AI_ASSISTANT_PERFUME.ipynb`):
   * **Data Ingestion:** Loads a dataset (`fra_perfumes.csv`) containing over 70,000 unique fragrance entries.
   * **Feature Extraction:** Concatenates structural columns into a singular semantic feature token string:
     $$\text{features} = \text{Main Accords} + \text{" "} + \text{Gender}$$
   * **Vocabulary Fitting:** Fits a `TfidfVectorizer` to scale and weigh individual terms across the document corpus.
   * **Artifact Export:** Serializes the trained objects (`tfidf.pkl`), transformed sparse arrays (`vectors.pkl`), and the target tracking DataFrame (`data.pkl`) into binary formats via `pickle`.

2. Real-time Inference Phase (`app.py`):
   Vector Space Transformation: Converts raw user input text on-the-fly using the pre-trained TF-IDF model.
   Distance Calculation:** Measures the angular distance between the user input vector and all 70,103 dataset vectors using:
     $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
   Index Extraction: Identifies and sorts the indices of the top 5 highest-scoring rows, serving them directly to the interface.

---

### 📂 Project Structure
```text
├── AI_ASSISTANT_PERFUME.ipynb   # Jupyter Notebook for data cleaning, fitting, and serialization
├── app.py                       # Streamlit web application dashboard logic
├── fra_perfumes.csv             # Raw dataset containing 70,103 entries (omitted from tracking)
├── tfidf.pkl                    # Serialized TF-IDF Vectorizer instance
├── vectors.pkl                  # Serialized sparse matrix of pre-computed historical weights
├── data.pkl                     # Serialized Pandas DataFrame for interface rendering and lookups
└── requirements.txt             # Definitive list of pinned third-party Python environments

```
### 🛠️ Local Installation & Setup
Follow these structured instructions to build, configure, and launch this project environment locally on your workstation:

1. Clone the GitHub Repository
Bring down the project directory tree and source codes into your local environment using the clickable link below:

👉 Click Here to Open Repository

Bash
# Or copy and execute directly in your terminal:
git clone [https://github.com/AnzaKayani/Perfume-Recommendation-AI.git](https://github.com/AnzaKayani/Perfume-Recommendation-AI.git)
cd Perfume-Recommendation-AI


2. Set Up a Virtual Environment
Isolate your dependency tree to prevent library versioning conflicts across system packages:

Bash
# On Windows environments:
python -m venv venv
venv\Scripts\activate

# On macOS / Linux environments:
python3 -m venv venv
source venv/bin/activate

3. Install Third-Party Requirements
Install the mandatory collection of execution frameworks pinned within your requirements file:

Bash
pip install -r requirements.txt

4. Populate Source Models (Pickle Files)
Ensure that the source binary picklings are placed inside the root folder. If you need to re-generate them from your raw dataset, execute your preprocessing pipeline:

Bash
# Open and run all evaluation steps in your workspace notebook:
jupyter notebook AI_ASSISTANT_PERFUME.ipynb

5. Launch the Streamlit Web Application Server
Initialize the local web engine server. This automatically invokes a new interactive window inside your default internet browser:

Bash
streamlit run app.py


### 💻 Core Technology Stack
Web Interface Engine: Streamlit (v1.x)
Mathematical Vectorization: Scikit-Learn (TfidfVectorizer, cosine_similarity)
Structured Data Arrays: Pandas & NumPy
Object Serialization: Standard Python pickle Protocol
