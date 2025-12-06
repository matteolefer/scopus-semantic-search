import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
from collections import Counter
import networkx.algorithms.community as nx_comm # For community detection

# Ensure the analyze_with_gemini function is in utils.py
from utils import load_data, search_articles, analyze_with_gemini, validate_gemini_api_key

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Scopus AI Search", page_icon="🧬")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-card { background-color: #f9f9f9; padding: 10px; border-radius: 5px; text-align: center; }
    h3 { color: #0068c9; }
</style>
""", unsafe_allow_html=True)

def on_key_change():
    st.session_state.ready_to_validate = True

# --- SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")

    st.text_input(
        "Gemini API Key",
        type="password",
        help="Get a key at aistudio.google.com",
        key="api_key",
        on_change=on_key_change,   # validate only after typing stops
    )

    api_key = st.session_state.get("api_key", "")
    
    if not api_key:
        st.warning("⚠️ Enter your API Key to unlock AI features.")
        st.markdown("[🔑 Get Free API Key](https://aistudio.google.com/app/apikey)")
        st.session_state.api_valid = False

    else:
        if st.session_state.get("ready_to_validate", False):
            st.session_state.api_valid = validate_gemini_api_key(api_key)
            st.session_state.ready_to_validate = False   # reset

        if st.session_state.get("api_valid", False):
            st.success("✅ Valid API Key — AI Features Enabled")
        else:
            st.error("❌ Invalid API Key. Please enter a correct one.")
        
    st.markdown("---")
    st.markdown("👨‍💻 **Data Science Project**")
    st.markdown("Scopus Corpus Analysis & Semantic Search.")

# --- LOAD DATA ---
with st.spinner("Loading AI model and data..."):
    df, embeddings, model = load_data()

if df is None:
    st.stop() 

# --- CACHED FUNCTIONS (OPTIMIZATION) ---
@st.cache_data(show_spinner=False)
def get_unique_authors(df_input):
    """
    Extracts and sorts unique authors from the dataframe.
    Cached to avoid re-computation on every rerun.
    """
    all_authors_flat = [
        a.strip() 
        for sublist in df_input['authors'].dropna().astype(str).str.split(',') 
        for a in sublist
    ]
    auth_counts = Counter(all_authors_flat)
    # Return authors with >2 chars and not 'unknown', sorted by frequency
    return [a for a, c in auth_counts.most_common() if len(a) > 2 and "unknown" not in a.lower()]

# --- MAIN HEADER ---
st.title("🧬 Scopus Semantic Search Engine")
st.markdown("""
This application uses **Embeddings (SBERT)** and an **LLM (Gemini)** to explore a database of scientific articles.
""")

# --- TAB NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🔍 AI Search", "📊 Dashboard & Analytics", "🕸️ Author Network"])

# =========================================================
# TAB 1: SEMANTIC SEARCH (AI MODULE) 
# =========================================================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask your scientific question:", placeholder="Ex: Generative AI for energy efficiency...")
        # --- Reset AI state if the query changes ---
        if "last_query" not in st.session_state:
            st.session_state.last_query = ""

        if query != st.session_state.last_query:
            st.session_state.gemini_active = False
            st.session_state.ai_data = None
            st.session_state.top_results_for_ai = None
            st.session_state.last_query = query
    with col2:
        min_year = int(df['publication_year'].min())
        max_year = int(df['publication_year'].max())
        selected_years = st.slider("Filter by year", min_year, max_year, (min_year, max_year))

    if query:
        results = search_articles(query, model, embeddings, df, top_k=10)
        
        filtered_results = [
            res for res in results 
            if selected_years[0] <= res['year'] <= selected_years[1]
        ]
        
        st.markdown(f"### 🎯 Results found: {len(filtered_results)}")

        # Track AI analysis state
        if "gemini_active" not in st.session_state:
            st.session_state.gemini_active = False
        if "ai_data" not in st.session_state:
            st.session_state.ai_data = None
        if "top_results_for_ai" not in st.session_state:
            st.session_state.top_results_for_ai = None

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            if not st.session_state.gemini_active:
                run_ai = st.button("✨ Analyze with AI", type="primary", disabled=not st.session_state.api_valid)
                remove_ai = False
                sort_option = "Similarity Score"  # default when AI not run
            else:
                remove_ai = st.button("❌ Remove AI Analysis", type="secondary")
                run_ai = False

        if not api_key:
            st.caption("🔒 *Enter API Key in sidebar to unlock AI Reranking & Summarization*")

        # --- Remove AI Analysis ---
        if remove_ai:
            st.session_state.gemini_active = False
            st.session_state.ai_data = None
            st.session_state.top_results_for_ai = None
            st.success("AI analysis removed.")
            st.rerun()  

        # --- Run AI Analysis ---
        if run_ai and api_key:
            if len(filtered_results) > 0:
                with st.spinner("Gemini is reading the articles for you..."):
                    top_results = filtered_results[:5]
                    ai_response = analyze_with_gemini(query, top_results, api_key)
                    
                    try:
                        start_idx = ai_response.find('[')
                        end_idx = ai_response.rfind(']') + 1
                        
                        if start_idx != -1 and end_idx != 0:
                            clean_json = ai_response[start_idx:end_idx]
                            st.session_state.ai_data = json.loads(clean_json)
                            st.session_state.top_results_for_ai = top_results
                            st.session_state.gemini_active = True
                            st.rerun()  
                        else:
                            st.error("Gemini did not return a valid JSON list.")
                            st.code(ai_response)
                            
                    except json.JSONDecodeError:
                        st.error("Gemini response formatting error.")
                        st.code(ai_response)
            else:
                st.warning("No results to analyze.")

        # --- DISPLAY RESULTS ---
        if st.session_state.gemini_active and st.session_state.ai_data:
            st.success("✅ AI Analysis Complete")

            sort_option = st.radio(
                "Sort results by:", 
                ["AI Score", "Similarity Score"],
                index=0,
                horizontal=True
            )

            # Make sure top_results_for_ai is saved in session for re-use
            if "top_results_for_ai" not in st.session_state:
                st.session_state.top_results_for_ai = filtered_results[:5]

            combined = list(zip(st.session_state.top_results_for_ai, st.session_state.ai_data))

            # SORT based on dropdown
            if sort_option == "AI Score":
                combined.sort(key=lambda x: x[1].get('relevance_score', 0), reverse=True)
            # else: keep original order (Similarity Score)

            for idx, (res, ai) in enumerate(combined):
                score = ai.get('relevance_score', 0)
                
                if idx == 0 and score > 80:
                    st.balloons()

                short_title = (res['title'][:75] + '..') if len(res['title']) > 75 else res['title']
                
                with st.expander(f"🤖 AI Score: {score}/100 | {short_title}", expanded=(idx < 2)):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"### {res['title']}")
                        
                        doi_link = res['doi']
                        if doi_link and str(doi_link).lower() != 'nan' and str(doi_link) != '':
                            if not str(doi_link).startswith('http'):
                                doi_link = f"https://doi.org/{doi_link}"
                            st.caption(f"📅 **{res['year']}** | ✍️ {res['authors']} | 🔗 [Open Article]({doi_link})")
                        else:
                            st.caption(f"📅 **{res['year']}** | ✍️ {res['authors']}")
                        
                        st.markdown("---")
                        st.markdown("**📝 Summary:**")
                        st.info(ai.get('summary', 'No summary'))
                    
                    with c2:
                        st.markdown("#### 🔍 Why relevant?")
                        reason = ai.get('relevance_reason', '')
                        if score > 75:
                            st.success(f"**High Relevance**\n\n{reason}")
                        elif score > 50:
                            st.warning(f"**Partial Relevance**\n\n{reason}")
                        else:
                            st.error(f"**Low Relevance**\n\n{reason}")
                        st.metric("Confidence Score", f"{score}/100")
                        
                    st.markdown("---")
                    with st.expander("Read full abstract"):
                        st.write(res['abstract'])

            if len(filtered_results) > 5:
                st.markdown("### 📚 Other Results")
                for res in filtered_results[5:]:
                     with st.expander(f"📄 {res['score']:.2f} | {res['title']} ({res['year']})"):
                        st.markdown(f"### {res['title']}")
                        st.markdown(f"**✍️ Authors:** *{res['authors']}*")
                        doi_link = res['doi']
                        if doi_link and str(doi_link).lower() != 'nan' and str(doi_link) != '':
                             if not str(doi_link).startswith('http'):
                                doi_link = f"https://doi.org/{doi_link}"
                             st.markdown(f"🔗 [**Read Full Article**]({doi_link})")
                        st.markdown("---")
                        st.markdown(f"{res['abstract'][:300]}...") 

        else:
            if sort_option == "Similarity Score":
                filtered_results.sort(key=lambda x: x['score'], reverse=True)
            for res in filtered_results:
                short_title = (res['title'][:80] + '..') if len(res['title']) > 80 else res['title']
                with st.expander(f"📄 Score: {res['score']:.2f} | {short_title} ({res['year']})"):
                    st.markdown(f"### {res['title']}")
                    m1, m2 = st.columns([3, 1])
                    with m1:
                        st.markdown(f"**✍️ Authors:** {res['authors']}")
                        st.caption(f"🏢 {res['affiliation']}")
                    with m2:
                        doi_link = res['doi']
                        if doi_link and str(doi_link).lower() != 'nan' and str(doi_link) != '':
                            if not str(doi_link).startswith('http'):
                                doi_link = f"https://doi.org/{doi_link}"
                            st.success(f"🔗 [**Open Article**]({doi_link})")
                        else:
                            st.caption("No link available")
                    st.markdown("---")
                    st.markdown("**📖 Abstract:**")
                    st.write(res['abstract'])

# =========================================================
# TAB 2: VISUALIZATION DASHBOARD (VIZ MODULE)
# =========================================================
with tab2:
    st.header("📊 Interactive Dashboard")
    st.markdown("Explore the dataset with interactive charts. Use the filters below to customize the view.")

    # --- 1. FILTERS ---
    with st.expander("⚙️ **Global Filters**", expanded=True):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            # Year Range Filter
            min_y = int(df['publication_year'].min())
            max_y = int(df['publication_year'].max())
            year_range = st.slider("Select Year Range", min_y, max_y, (min_y, max_y), key="viz_year_slider")
        
        with f_col2:
            # Category Filter (if available)
            if 'category' in df.columns:
                # Split categories because one paper can have multiple "Comp Sci, Engineering"
                all_cats = df['category'].dropna().astype(str).str.split(', ').explode().unique()
                
                # Clean up categories for the filter list (remove Unknown)
                clean_cats = [c for c in all_cats if c.lower() not in ['unknown', 'nan', '']]
                selected_cats = st.multiselect("Filter by Subject Area", options=sorted(clean_cats))
            else:
                selected_cats = []

    # Apply Filters to DataFrame
    df_viz = df[(df['publication_year'] >= year_range[0]) & (df['publication_year'] <= year_range[1])].copy()
    
    if selected_cats:
        pattern = '|'.join(selected_cats)
        df_viz = df_viz[df_viz['category'].astype(str).str.contains(pattern, na=False)]

    # --- 2. KEY METRICS (KPIs) ---
    st.markdown("#### 🚀 Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("📚 Total Articles", f"{len(df_viz):,}")
    kpi2.metric("📅 Time Span", f"{year_range[0]} - {year_range[1]}")
    
    unique_authors = len(set([a.strip() for sublist in df_viz['authors'].astype(str).str.split(',') for a in sublist]))
    kpi3.metric("✍️ Active Researchers", f"{unique_authors:,}")
    
    unique_affiliations = df_viz['affiliation'].nunique()
    kpi4.metric("🏫 Institutions", f"{unique_affiliations:,}")

    st.markdown("---")

    # --- 3. VISUALIZATIONS ROW 1 ---
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        st.subheader("📈 Publications Trend")
        pub_trend = df_viz.groupby('publication_year').size().reset_index(name='Count')
        fig_trend = px.line(pub_trend, x='publication_year', y='Count', markers=True, 
                            title="Number of Articles per Year", template="plotly_white")
        fig_trend.update_xaxes(type='category')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with row1_2:
        st.subheader("🧩 Subject Area Distribution")
        if 'category' in df_viz.columns:
            cats_series = df_viz['category'].dropna().astype(str).str.split(', ').explode()
            cats_series = cats_series[~cats_series.str.lower().isin(['unknown', 'nan', ''])]
            top_cats = cats_series.value_counts().head(10).reset_index()
            top_cats.columns = ['Subject Area', 'Count']
            
            fig_pie = px.pie(top_cats, values='Count', names='Subject Area', hole=0.4,
                             title="Top 10 Subject Areas", template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No Category data available.")

    # --- 4. VISUALIZATIONS ROW 2 ---
    row2_1, row2_2 = st.columns(2)
    
    with row2_1:
        st.subheader("🏫 Top 10 Affiliations")
        if 'affiliation' in df_viz.columns:
            aff_series = df_viz['affiliation'].fillna("Unknown")
            top_aff = aff_series.value_counts().head(10).reset_index()
            top_aff.columns = ['Affiliation', 'Count']
            
            fig_bar_aff = px.bar(top_aff, x='Count', y='Affiliation', orientation='h', 
                                 title="Most Productive Institutions", template="plotly_white", color='Count')
            fig_bar_aff.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_aff, use_container_width=True)

    with row2_2:
        st.subheader("✍️ Top 10 Active Authors")
        all_authors = [a.strip() for sublist in df_viz['authors'].dropna().astype(str).str.split(',') for a in sublist]
        all_authors = [a for a in all_authors if a.lower() not in ['unknown', 'unknown author', '']]
        
        if all_authors:
            top_authors_count = Counter(all_authors).most_common(10)
            df_auth = pd.DataFrame(top_authors_count, columns=['Author', 'Count'])
            fig_bar_auth = px.bar(df_auth, x='Count', y='Author', orientation='h',
                                  title="Most Frequent Authors", template="plotly_white", color='Count')
            fig_bar_auth.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_auth, use_container_width=True)
        else:
            st.info("No Author data available.")
            
    # --- 5. VISUALIZATIONS ROW 3 (ADVANCED) ---
    st.subheader("🌡️ Research Hotspots (Heatmap)")
    if 'category' in df_viz.columns:
        df_exploded = df_viz[['publication_year', 'category']].dropna()
        df_exploded['category'] = df_exploded['category'].astype(str).str.split(', ')
        df_exploded = df_exploded.explode('category')
        df_exploded = df_exploded[~df_exploded['category'].str.lower().isin(['unknown', 'nan', ''])]
        
        top_15_cats = df_exploded['category'].value_counts().head(15).index
        df_heatmap = df_exploded[df_exploded['category'].isin(top_15_cats)]
        
        heatmap_data = df_heatmap.groupby(['category', 'publication_year']).size().reset_index(name='Count')
        
        fig_heat = px.density_heatmap(heatmap_data, x='publication_year', y='category', z='Count',
                                      title="Intensity of Research by Topic & Year", 
                                      color_continuous_scale="Viridis", template="plotly_white")
        fig_heat.update_xaxes(type='category')
        st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 3: NETWORK ANALYSIS (STATIC STAR GRAPH - NETWORKX)
# =========================================================
with tab3:
    st.header("🕸️ Researcher Collaboration Network (Star View)")
    st.markdown("Visualize the direct collaborators of a researcher. Each node represents an author, and edges represent co-authorship links.")

    # 1. PREPARE AUTHOR LIST
    with st.spinner("Preparing author list..."):
        sorted_authors = get_unique_authors(df)

    # 2. SELECTION UI
    col_sel, col_net_info = st.columns([1, 2])
    with col_sel:

        selected_author = st.selectbox(
            "🔍 Select Researcher",
            sorted_authors[:3000],
            index=0,
            help="Type to search"
        )

        st.caption("🔧 Display Controls")

        min_collab = st.slider("Min. Shared Papers", 1, 10, 1)
        max_nodes = st.slider("Max Collaborators to Show", 10, 100, 30)

    # 3. BUILD GRAPH LOGIC
    if selected_author:

        # Filter rows containing this author
        author_papers = df[df["authors"].astype(str).str.contains(selected_author, regex=False, na=False)]

        # Extract all co-authors
        all_co_authors = []
        for _, row in author_papers.iterrows():
            paper_authors = [a.strip() for a in str(row["authors"]).split(",")]
            valid_authors = [a for a in paper_authors if a != selected_author and len(a) > 2]
            all_co_authors.extend(valid_authors)

        # Count collaborations
        co_auth_counts = Counter(all_co_authors)

        filtered_co_authors = {k: v for k, v in co_auth_counts.items() if v >= min_collab}
        top_co_authors = sorted(filtered_co_authors.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

        with col_net_info:
            st.info(
                f"Researcher **{selected_author}** has **{len(filtered_co_authors)}** collaborators "
                f"(≥ {min_collab} shared papers). Showing top **{len(top_co_authors)}**."
            )

        # --- 3.2 BUILD STATIC GRAPH (NETWORKX) ---
        G = nx.Graph()

        # Add central author
        G.add_node(selected_author)

        # Add collaborators
        for co_auth, weight in top_co_authors:
            G.add_node(co_auth)
            G.add_edge(selected_author, co_auth, weight=weight)

        # ---- 4. STATIC VISUALIZATION ----
        st.subheader("📌 Collaboration Graph")

        # Better-looking static visualization (no background)
        plt.figure(figsize=(10, 10), facecolor="none")

        # Improved spacing between nodes
        pos = nx.spring_layout(G, seed=42, k=1.3)

        # Node styling
        node_colors = []
        for node in G.nodes():
            if node == selected_author:
                node_colors.append("#ff6b6b")   # Highlight main author
            else:
                node_colors.append("#4a90e2")   # Blue for collaborators

        # Increase node sizes for readability
        node_sizes_improved = []
        for node in G.nodes():
            if node == selected_author:
                node_sizes_improved.append(2500)   
            else:
                node_sizes_improved.append(900)

        # Remove axes background
        ax = plt.gca()
        ax.set_facecolor("none")

        # Draw nodes & edges
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes_improved,
            node_color=node_colors,
            edgecolors="black",
            linewidths=1.2
        )

        nx.draw_networkx_edges(
            G, pos,
            width=1.4,
            alpha=0.5,
            edge_color="gray"
        )

        # Label styling (white pill behind text)
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_color="black",
            font_weight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.7
            )
        )

        plt.axis("off")
        plt.tight_layout()

        st.pyplot(plt)

        # --- 5. DATA TABLE ---
        with st.expander("📊 View Collaboration Details", expanded=True):
            if top_co_authors:
                df_collab = pd.DataFrame(top_co_authors, columns=["Co-Author", "Shared Papers"])
                st.dataframe(df_collab, use_container_width=True)
            else:
                st.info("No collaborators found with current filters.")
