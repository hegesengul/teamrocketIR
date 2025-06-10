import streamlit as st
from BM25.bm25 import search_code_snippet as bm25_search

st.set_page_config(page_title="Code Search Engine", layout="wide")
st.title("🔍 Code Snippet Similarity Search")

model_option = st.selectbox("Select Retrieval Model:", ["BM25-CodeBERTa-small-v1", "BM25-Roberta"])

query = st.text_area("Enter your code snippet:", height=200)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid code snippet.")
    else:
        with st.spinner("Searching..."):
            results = bm25_search(query, model_option, top_k=5)

        st.subheader("🔎 Search Results")
        for idx, row in results.iterrows():
            st.markdown(f"**Rank {idx + 1}**")
            st.markdown(f"**Distance**: {row['euclidean_distance']:.4f}")
            st.code(row["originalCode"], language=row["language"])
            st.markdown("---")

# TODO test