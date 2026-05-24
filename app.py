import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Saved Files
# -------------------------
@st.cache_resource
def load_files():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    with open("data.pkl", "rb") as f:
        df = pickle.load(f)

    return tfidf, vectors, df

tfidf, vectors, df = load_files()

# -------------------------
# Page Settings
# -------------------------
st.set_page_config(page_title = "AI Assistant Perfume ", page_icon="💐")

# -------------------------
# UI
# -------------------------
st.title("💐 Perfume Recommendation AI")
st.write("Find perfumes based on fragrance type")

st.markdown("### 🔍 Search Your Fragrance")

user_input = st.text_input(
    "Enter fragrance (e.g. citrus, woody, floral):"
)

# -------------------------
# Recommendation Function
# -------------------------
def recommend_perfume(query):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, vectors)

    top_indexes = scores[0].argsort()[-5:][::-1]
    return df.iloc[top_indexes]

# -------------------------
# Button Action
# -------------------------
if st.button("✨ Recommend"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a fragrance")
    else:
        results = recommend_perfume(user_input)

        st.markdown("## ✨ Recommended Perfumes")

        for _, row in results.iterrows():
            st.markdown(f"### 🌸 {row['Name']}")
            st.write(f"**Accords:** {row['Main Accords']}")
            st.write("---")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with using streamlit")