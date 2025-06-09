import streamlit as st
from BM25.bm25 import search_code_snippet as bm25_search

st.set_page_config(page_title="Code Search Engine", layout="wide")
st.title("🔍 Code Snippet Similarity Search")

model_option = st.selectbox("Select Retrieval Model:", ["BM25", "CodeBERT", "GraphCodeBERT"])
language_option = st.selectbox("Select Programming Language:", ["Python", "Java"])

query = st.text_area("Enter your code snippet:", height=200)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid code snippet.")
    else:
        with st.spinner("Searching..."):
            if model_option == "BM25":
                results = bm25_search(query, language=language_option, top_k=5)
            #elif model_option == "CodeBERT":
            #   results = codebert_search(query, top_k=10, language=language_option)
            #elif model_option == "GraphCodeBERT":
            #   results = graphcodebert_search(query, top_k=10, language=language_option)

        st.subheader("🔎 Search Results")
        for idx, row in results.iterrows():
            st.markdown(f"**Rank {idx + 1}**")
            st.markdown(f"**Similarity Score**: {row['dense_score']:.4f}")
            st.code(row["originalCode"], language=language_option.lower())
            st.markdown("---")