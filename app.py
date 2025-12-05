import streamlit as st
import pandas as pd
import plotly.express as px
import json
# Ensure the analyze_with_gemini function is in utils.py
from utils import load_data, search_articles, analyze_with_gemini

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Scopus AI Search", page_icon="🧬")

# --- CUSTOM CSS (Optional, for styling) ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .reportview-container { background: #fdfdfd; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Enter your Google Gemini API key to enable intelligent article analysis.")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a key at aistudio.google.com")
    st.markdown("---")
    st.markdown("👨‍💻 **Data Science Project**")
    st.markdown("Scopus Corpus Analysis & Semantic Search.")

# --- LOAD DATA ---
# Using the cache defined in utils.py to avoid reloading on every interaction
with st.spinner("Loading AI model and data..."):
    df, embeddings, model = load_data()

if df is None:
    st.stop() # Stop the app if data fails to load

# --- MAIN HEADER ---
st.title("🧬 Scopus Semantic Search Engine")
st.markdown("""
This application uses **Embeddings (SBERT)** and an **LLM (Gemini)** to explore a database of scientific articles.
""")

# --- TAB NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🔍 AI Search", "📊 Statistics & Trends", "🕸️ Author Network"])

# =========================================================
# TAB 1: SEMANTIC SEARCH (AI MODULE)
# =========================================================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask your scientific question:", placeholder="Ex: Generative AI for energy efficiency...")
    with col2:
        # Filter by year
        min_year = int(df['publication_year'].min())
        max_year = int(df['publication_year'].max())
        selected_years = st.slider("Filter by year", min_year, max_year, (min_year, max_year))

    if query:
        # 1. Vector Search (Fast)
        # Retrieve a few more results (10) to give choices
        results = search_articles(query, model, embeddings, df, top_k=10)
        
        # Manual filtering by year on the returned results
        filtered_results = [
            res for res in results 
            if selected_years[0] <= res['year'] <= selected_years[1]
        ]
        
        st.markdown(f"### 🎯 Results found: {len(filtered_results)}")

        # --- GEMINI LOGIC ---
        gemini_active = False
        
        if api_key:
            # If we have a key, show the magic button
            if st.button("✨ Analyze relevance with Gemini (AI)", type="primary"):
                if len(filtered_results) > 0:
                    with st.spinner("Gemini is reading the articles for you..."):
                        # Send only the top 5 to avoid saturating the API
                        top_results_for_ai = filtered_results[:5]
                        ai_response = analyze_with_gemini(query, top_results_for_ai, api_key)
                        
                        # Clean JSON returned by Gemini
                        clean_json = ai_response.replace('```json', '').replace('```', '').strip()
                        
                        try:
                            ai_data = json.loads(clean_json)
                            gemini_active = True # Switch to AI display mode
                        except json.JSONDecodeError:
                            st.error("Gemini response formatting error. Standard display.")
                            st.code(ai_response) # Debug
                else:
                    st.warning("No results to analyze in this year range.")

        # --- DISPLAY RESULTS ---
        
        if gemini_active and 'ai_data' in locals():
            # MODE 1: AI-ENRICHED DISPLAY
            st.success("AI analysis generated successfully!")
            
            # Loop through analyzed results
            for idx, (res, ai) in enumerate(zip(top_results_for_ai, ai_data)):
                with st.expander(f"🤖 {ai.get('relevance_score', 0)}% | {res['title']} ({res['year']})", expanded=True):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**Summary:** {ai.get('summary', 'No summary')}")
                        st.markdown(f"**Analysis:** {ai.get('relevance_reason', 'No analysis')}")
                        st.caption(f"Authors: {res['authors']} | DOI: {res['doi']}")
                        st.info(f"Abstract excerpt: {res['abstract'][:300]}...")
                    
                    with col_b:
                        score = ai.get('relevance_score', 0)
                        st.metric("Relevance", f"{score}/100")
                        if score > 75:
                            st.balloons() if idx == 0 else None # Little fun for the top 1
            
            if len(filtered_results) > 5:
                st.markdown("---")
                st.caption("Other results (Not analyzed by AI):")
                # Display the rest in standard mode
                for res in filtered_results[5:]:
                     with st.expander(f"{res['score']:.2f} | {res['title']} ({res['year']})"):
                        st.write(res['abstract'])

        else:
            # MODE 2: STANDARD DISPLAY (No key or not clicked yet)
            if not api_key:
                st.info("💡 Tip: Add a Gemini API key in the sidebar to get automatic summaries.")
            
            for res in filtered_results:
                with st.expander(f"📄 Score: {res['score']:.2f} | {res['title']} ({res['year']})"):
                    st.markdown(f"**Authors:** {res['authors']}")
                    st.markdown(f"**Affiliation:** {res['affiliation']}")
                    st.write(res['abstract'])
                    if res['doi']:
                        st.markdown(f"🔗 [Link to article](https://doi.org/{res['doi']})")

# =========================================================
# TAB 2: VISUALIZATION (VIZ MODULE)
# =========================================================
with tab2:
    st.header("📊 Exploratory Data Analysis")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # 1. Publications by year chart
        st.subheader("📈 Temporal Evolution")
        pub_counts = df['publication_year'].value_counts().sort_index().reset_index()
        pub_counts.columns = ['Year', 'Number of Articles']
        fig_line = px.line(pub_counts, x='Year', y='Number of Articles', markers=True, title="Articles per Year")
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col_viz2:
        # 2. Top Affiliations (Bar Chart)
        st.subheader("🏫 Top Affiliations")
        # Quick cleanup for display
        if 'affiliation' in df.columns:
            top_aff = df['affiliation'].fillna("Unknown").astype(str).value_counts().head(10).reset_index()
            top_aff.columns = ['Affiliation', 'Count']
            fig_bar = px.bar(top_aff, x='Count', y='Affiliation', orientation='h', color='Count', title="Most Active Institutions")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No affiliation data available.")

# =========================================================
# TAB 3: NETWORK (PLACEHOLDER)
# =========================================================
with tab3:
    st.header("🕸️ Co-author Network")
    st.info("🚧 This module allows exploring collaborations between researchers.")
    st.markdown("""
    **Future implementation idea:**
    1. Select an author.
    2. Use `networkx` to find their co-authors in the dataset.
    3. Visualize the graph with `streamlit-agraph` or `pyvis`.
    """)
    
    # Visual placeholder to show the professor we planned this
    st.image("https://raw.githubusercontent.com/WestHealth/pyvis/master/docs/images/network_example.png", caption="Example of expected PyVis output", width=600)