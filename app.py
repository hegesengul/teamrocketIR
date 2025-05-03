import streamlit as st
from BM25.bm25 import search_code_snippet

st.title("🔍 Code Search with BM25")

query = st.text_area("Enter your code snippet:", height=200)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a code snippet")
    else:
        with st.spinner("Aranıyor..."):
            results = search_code_snippet(query, top_k=10)

        st.subheader("🔎 Search Results")
        for idx, row in results.iterrows():
            st.markdown(f"**Rank {idx + 1}:**")
            st.markdown(f"**Document ID**: {row['docno']}")
            st.markdown(f"**Similarity Score**: {row['score']:.4f}")
            
            st.code(row["text"], language="python")
            st.markdown("---")
