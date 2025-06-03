import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import json
from pathlib import Path
import time
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Alzheimer Classification | DenseNet Hyperparameter Optimization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Hacker Theme CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background-color: #000000;
        color: #00ff00;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        background-color: #000000;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #111111;
        border-right: 2px solid #00ff00;
    }
    
    .css-1d391kg .css-1v3fvcr {
        background-color: #111111;
        color: #00ff00;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #00ff00;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00;
        font-family: 'Courier New', monospace;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00, 0 0 15px #00ff00; }
        to { text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
    }
    
    /* Folder cards */
    .folder-card {
        background: linear-gradient(135deg, #001100 0%, #002200 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #00ff00;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        font-family: 'Courier New', monospace;
    }
    
    .folder-card h4 {
        color: #00ff00;
        text-shadow: 0 0 5px #00ff00;
        margin-bottom: 1rem;
    }
    
    .folder-card p {
        color: #00cc00;
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-completed {
        color: #00ff00;
        font-weight: bold;
        text-shadow: 0 0 5px #00ff00;
    }
    .status-running {
        color: #ffff00;
        font-weight: bold;
        text-shadow: 0 0 5px #ffff00;
        animation: blink 1s infinite;
    }
    .status-error {
        color: #ff0000;
        font-weight: bold;
        text-shadow: 0 0 5px #ff0000;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.5; }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #001100 0%, #002200 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00aa00;
        margin: 0.5rem 0;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
    }
    
    /* Results section */
    .results-section {
        background: linear-gradient(135deg, #001100 0%, #002200 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #00ff00;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Streamlit components styling */
    .stSelectbox > div > div {
        background-color: #111111;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    .stSelectbox > div > div > div {
        color: #00ff00;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #003300 0%, #006600 100%);
        color: #00ff00;
        border: 2px solid #00ff00;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #006600 0%, #009900 100%);
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
        transform: translateY(-2px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111111;
        border-bottom: 2px solid #00ff00;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #111111;
        color: #00ff00;
        border: 1px solid #00aa00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #002200;
        color: #00ff00;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #003300 0%, #006600 100%);
        color: #00ff00;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: #111111;
        border: 1px solid #00ff00;
    }
    
    .stDataFrame table {
        background-color: #111111;
        color: #00ff00;
    }
    
    .stDataFrame th {
        background-color: #002200;
        color: #00ff00;
        border: 1px solid #00aa00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    .stDataFrame td {
        background-color: #111111;
        color: #00cc00;
        border: 1px solid #003300;
    }
    
    /* Text areas */
    .stTextArea textarea {
        background-color: #111111;
        color: #00ff00;
        border: 2px solid #00aa00;
        font-family: 'Courier New', monospace;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: linear-gradient(135deg, #001100 0%, #002200 100%);
        border: 1px solid #00aa00;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .css-1xarl3l p {
        color: #00ff00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #111111;
        color: #00ff00;
        border: 1px solid #00aa00;
        font-family: 'Courier New', monospace;
    }
    
    .streamlit-expanderContent {
        background-color: #111111;
        border: 1px solid #00aa00;
        color: #00cc00;
    }
    
    /* Sidebar text */
    .css-1d391kg .css-1v3fvcr h2,
    .css-1d391kg .css-1v3fvcr h3,
    .css-1d391kg .css-1v3fvcr p {
        color: #00ff00;
        font-family: 'Courier New', monospace;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #002200;
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    
    .stError {
        background-color: #220000;
        border: 1px solid #ff0000;
        color: #ff0000;
    }
    
    .stWarning {
        background-color: #222200;
        border: 1px solid #ffff00;
        color: #ffff00;
    }
    
    .stInfo {
        background-color: #002222;
        border: 1px solid #00ffff;
        color: #00ffff;
    }
    
    /* Spinner */
    .stSpinner {
        color: #00ff00;
    }
    
    /* Charts */
    .js-plotly-plot {
        background-color: #111111;
    }
    
    /* Terminal-like effect for code blocks */
    .stCode {
        background-color: #000000;
        border: 2px solid #00ff00;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        background-color: #111111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #003300 0%, #006600 100%);
        border-radius: 5px;
        border: 1px solid #00ff00;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #006600 0%, #009900 100%);
    }
    
    /* Matrix-like background effect */
    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(90deg, transparent 24%, rgba(0, 255, 0, 0.03) 25%, rgba(0, 255, 0, 0.03) 26%, transparent 27%, transparent 74%, rgba(0, 255, 0, 0.03) 75%, rgba(0, 255, 0, 0.03) 76%, transparent 77%, transparent),
            linear-gradient(transparent 24%, rgba(0, 255, 0, 0.03) 25%, rgba(0, 255, 0, 0.03) 26%, transparent 27%, transparent 74%, rgba(0, 255, 0, 0.03) 75%, rgba(0, 255, 0, 0.03) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

# Title with enhanced hacker styling
st.markdown('<h1 class="main-header">🧠 ALZHEIMER CLASSIFICATION NEURAL NETWORK</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">DenseNet | Optuna Hyperparameter Optimization</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 SYSTEM CONFIGURATION")
st.sidebar.markdown("---")

# Model directories
model_dirs = ["DenseNet121", "DenseNet169", "DenseNet201"]
available_dirs = []

# Check which directories exist
for model_dir in model_dirs:
    if os.path.exists(model_dir):
        available_dirs.append(model_dir)

if not available_dirs:
    st.error("❌ ERROR: No neural network directories detected!")
    st.info("Required directories: DenseNet121, DenseNet169, DenseNet201")
    st.stop()

# Sidebar options
st.sidebar.subheader("🎯 NEURAL NETWORK SELECTION")
selected_model = st.sidebar.selectbox(
    "SELECT TARGET MODEL:",
    available_dirs,
    help="Choose which DenseNet model to analyze"
)

# Initialize session state
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}

# Function to check what files exist in a directory
def get_directory_files(model_dir):
    """Get all files in the model directory"""
    files = {}
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isfile(item_path):
                files[item] = item_path
    return files

# Function to read best hyperparameters
def read_best_hyperparameters(model_dir):
    """Read best hyperparameters from text file"""
    hyperparam_file = os.path.join(model_dir, "best_hyperparameters.txt")
    if os.path.exists(hyperparam_file):
        try:
            with open(hyperparam_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except:
            return None
    return None

# Function to load CSV data
def load_csv_data(file_path):
    """Load CSV file and return DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"ERROR loading {file_path}: {e}")
        return None

# Function to display image
def display_image(image_path, caption=""):
    """Display image with caption"""
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=caption, use_container_width=True)
            return True
    except Exception as e:
        st.error(f"ERROR loading image {image_path}: {e}")
    return False

# Function to convert notebook to python (simplified)
def convert_notebook_to_python(notebook_path, output_path):
    """Convert notebook to Python using simple parsing"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import json
        nb_data = json.loads(content)
        
        python_code = "#!/usr/bin/env python\n"
        python_code += "# Auto-converted from Jupyter notebook\n\n"
        
        if 'cells' in nb_data:
            for i, cell in enumerate(nb_data['cells']):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        source = ''.join(source)
                    
                    if source.strip():
                        python_code += f"# Cell {i+1}\n"
                        python_code += source
                        if not source.endswith('\n'):
                            python_code += '\n'
                        python_code += '\n'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        
        return True, python_code
        
    except Exception as e:
        return False, str(e)

# Main content
st.header(f"🚀 {selected_model} NEURAL NETWORK ANALYSIS DASHBOARD")

# Get files in selected directory
model_files = get_directory_files(selected_model)

# Display overview
col1, col2, col3 = st.columns(3)

with col1:
    csv_files = [f for f in model_files.keys() if f.endswith('.csv')]
    st.metric("📊 DATA FILES", len(csv_files))

with col2:
    png_files = [f for f in model_files.keys() if f.endswith('.png')]
    st.metric("📈 VISUALIZATIONS", len(png_files))

with col3:
    has_notebook = 'main.ipynb' in model_files
    st.metric("📓 JUPYTER NOTEBOOK", "ONLINE" if has_notebook else "OFFLINE")

# Status card with hacker styling
st.markdown(f"""
<div class="folder-card">
    <h4>📁 {selected_model} SYSTEM STATUS</h4>
    <p style="color:white"><strong>[DIRECTORY]:</strong> {selected_model}/</p>
    <p><strong>[TOTAL_FILES]:</strong> {len(model_files)}</p>
    <p><strong>[RESULTS_STATUS]:</strong> {"✅ AVAILABLE" if len(csv_files) > 0 else "❌ NOT_FOUND"}</p>
    <p><strong>[VISUALIZATION_STATUS]:</strong> {"✅ AVAILABLE" if len(png_files) > 0 else "❌ NOT_FOUND"}</p>
    <p><strong>[SYSTEM_READY]:</strong> {"✅ TRUE" if len(model_files) > 0 else "❌ FALSE"}</p>
</div>
""", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["OPTIMAL_RESULTS", "METRICS_ANALYSIS", "DATA_VISUALIZATION", "FILE_SYSTEM", "NOTEBOOK_CONTROL"])

with tab1:
    st.header("🏆 OPTIMAL HYPERPARAMETERS & PERFORMANCE METRICS")
    
    # Load best hyperparameters
    best_params = read_best_hyperparameters(selected_model)
    if best_params:
        st.subheader("📝 OPTIMAL HYPERPARAMETER CONFIGURATION")
        st.text_area("HYPERPARAMETERS:", best_params, height=200)
    else:
        st.info("WARNING: best_hyperparameters.txt NOT FOUND")
    
    # Show key metrics files
    col1, col2 = st.columns(2)
    
    with col1:
        # Best hybrid metrics
        best_hybrid_file = os.path.join(selected_model, "best_hybrid_per_class_metrics.csv")
        if os.path.exists(best_hybrid_file):
            st.subheader("🎯 HYBRID MODEL PERFORMANCE")
            df_hybrid = load_csv_data(best_hybrid_file)
            if df_hybrid is not None:
                st.dataframe(df_hybrid, use_container_width=True)
    
    with col2:
        # DenseNet metrics
        densenet_metrics_file = os.path.join(selected_model, "densenet_per_class_metrics.csv")
        if os.path.exists(densenet_metrics_file):
            st.subheader("🧠 DENSENET MODEL PERFORMANCE")
            df_densenet = load_csv_data(densenet_metrics_file)
            if df_densenet is not None:
                st.dataframe(df_densenet, use_container_width=True)
    
    # ROC AUC Comparison
    roc_file = os.path.join(selected_model, "roc_auc_comparison.csv")
    if os.path.exists(roc_file):
        st.subheader("📈 ROC AUC PERFORMANCE COMPARISON")
        df_roc = load_csv_data(roc_file)
        if df_roc is not None:
            st.dataframe(df_roc, use_container_width=True)
            
            # Create a simple bar chart if possible
            if len(df_roc.columns) >= 2:
                try:
                    chart_data = df_roc.set_index(df_roc.columns[0])
                    st.bar_chart(chart_data)
                except:
                    pass

with tab2:
    st.header("📊 DETAILED PERFORMANCE METRICS ANALYSIS")
    
    # All models metrics
    all_models_file = os.path.join(selected_model, "all_models_metrics.csv")
    if os.path.exists(all_models_file):
        st.subheader("🔍 COMPREHENSIVE MODEL COMPARISON")
        df_all = load_csv_data(all_models_file)
        if df_all is not None:
            st.dataframe(df_all, use_container_width=True)
            
            # Show summary statistics
            if not df_all.empty:
                st.subheader("📈 STATISTICAL ANALYSIS")
                numeric_cols = df_all.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df_all[numeric_cols].describe(), use_container_width=True)
    
    # SVM Optimization Results
    svm_file = os.path.join(selected_model, "svm_optimization_results.csv")
    if os.path.exists(svm_file):
        st.subheader("⚙️ SVM OPTIMIZATION RESULTS")
        df_svm = load_csv_data(svm_file)
        if df_svm is not None:
            st.dataframe(df_svm.head(20), use_container_width=True)
            
            # Show best SVM results
            if 'score' in df_svm.columns or 'accuracy' in df_svm.columns:
                score_col = 'score' if 'score' in df_svm.columns else 'accuracy'
                best_svm = df_svm.loc[df_svm[score_col].idxmax()]
                st.subheader("🏅 OPTIMAL SVM CONFIGURATION")
                st.json(best_svm.to_dict())

with tab3:
    st.header("📈 NEURAL NETWORK VISUALIZATION MATRIX")
    
    # Display all PNG files
    png_files = [f for f in model_files.keys() if f.endswith('.png')]
    
    if png_files:
        # Organize visualizations by category
        confusion_matrices = [f for f in png_files if 'confusion' in f.lower()]
        training_plots = [f for f in png_files if 'training' in f.lower() or 'history' in f.lower()]
        comparison_plots = [f for f in png_files if 'comparison' in f.lower()]
        roc_plots = [f for f in png_files if 'roc' in f.lower()]
        other_plots = [f for f in png_files if f not in confusion_matrices + training_plots + comparison_plots + roc_plots]
        
        if confusion_matrices:
            st.subheader("🎯 CONFUSION MATRIX ANALYSIS")
            cols = st.columns(min(2, len(confusion_matrices)))
            for i, plot_file in enumerate(confusion_matrices):
                with cols[i % 2]:
                    display_image(model_files[plot_file], f"{plot_file.replace('.png', '').replace('_', ' ').upper()}")
        
        if training_plots:
            st.subheader("📚 TRAINING HISTORY ANALYSIS")
            for plot_file in training_plots:
                display_image(model_files[plot_file], f"{plot_file.replace('.png', '').replace('_', ' ').upper()}")
        
        if comparison_plots:
            st.subheader("⚖️ MODEL COMPARISON ANALYSIS")
            cols = st.columns(min(2, len(comparison_plots)))
            for i, plot_file in enumerate(comparison_plots):
                with cols[i % 2]:
                    display_image(model_files[plot_file], f"{plot_file.replace('.png', '').replace('_', ' ').upper()}")
        
        if roc_plots:
            st.subheader("📈 ROC CURVE ANALYSIS")
            for plot_file in roc_plots:
                display_image(model_files[plot_file], f"{plot_file.replace('.png', '').replace('_', ' ').upper()}")
        
        if other_plots:
            st.subheader("📊 ADDITIONAL VISUALIZATIONS")
            cols = st.columns(min(2, len(other_plots)))
            for i, plot_file in enumerate(other_plots):
                with cols[i % 2]:
                    display_image(model_files[plot_file], f"{plot_file.replace('.png', '').replace('_', ' ').upper()}")
    else:
        st.info("WARNING: No visualization files (.png) detected in system")

with tab4:
    st.header("📋 FILE SYSTEM BROWSER")
    
    # Show all files in directory
    st.subheader(f"📁 FILES IN {selected_model}/ DIRECTORY")
    
    file_types = {}
    for filename, filepath in model_files.items():
        ext = filename.split('.')[-1] if '.' in filename else 'no extension'
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(filename)
    
    for ext, files in file_types.items():
        with st.expander(f"📄 {ext.upper()} FILES ({len(files)})"):
            for filename in files:
                filepath = model_files[filename]
                st.write(f"**{filename}**")
                
                if filename.endswith('.csv'):
                    if st.button(f"📊 LOAD_DATA: {filename}", key=f"view_{filename}"):
                        df = load_csv_data(filepath)
                        if df is not None:
                            st.dataframe(df, use_container_width=True)
                
                elif filename.endswith('.txt') or filename.endswith('.md'):
                    if st.button(f"📝 READ_FILE: {filename}", key=f"view_{filename}"):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            st.text_area(f"CONTENT OF {filename}", content, height=300)
                        except Exception as e:
                            st.error(f"ERROR reading {filename}: {e}")
                
                elif filename.endswith('.png'):
                    if st.button(f"🖼️ DISPLAY: {filename}", key=f"view_{filename}"):
                        display_image(filepath, f"{filename}")

with tab5:
    st.header("🔧 JUPYTER NOTEBOOK CONTROL PANEL")
    
    # Check if main.ipynb exists
    notebook_path = os.path.join(selected_model, "main.ipynb")
    
    if os.path.exists(notebook_path):
        st.success("✅ NOTEBOOK STATUS: ONLINE")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 CONVERT_TO_PYTHON"):
                python_path = os.path.join(selected_model, "main_converted.py")
                
                with st.spinner("CONVERTING NOTEBOOK..."):
                    success, result = convert_notebook_to_python(notebook_path, python_path)
                    
                    if success:
                        st.success("✅ CONVERSION SUCCESSFUL")
                        st.text_area("PYTHON CODE PREVIEW", result[:500] + "..." if len(result) > 500 else result, height=200)
                    else:
                        st.error(f"❌ CONVERSION FAILED: {result}")
        
        with col2:
            if st.button("🚀 EXECUTE_OPTIMIZATION"):
                python_path = os.path.join(selected_model, "main_converted.py")
                if os.path.exists(python_path):
                    st.session_state.execution_status[selected_model] = "Running"
                    st.rerun()
                else:
                    st.error("❌ ERROR: Convert notebook to Python first!")
        
        with col3:
            if st.button("📓 VIEW_NOTEBOOK_DATA"):
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Try to parse notebook JSON
                    try:
                        nb_data = json.loads(content)
                        if 'cells' in nb_data:
                            st.write(f"📊 NOTEBOOK CONTAINS {len(nb_data['cells'])} CELLS")
                            
                            # Show first few code cells
                            code_cells = [cell for cell in nb_data['cells'] if cell.get('cell_type') == 'code']
                            if code_cells:
                                st.write(f"🔍 DETECTED {len(code_cells)} CODE CELLS")
                                st.code(str(code_cells[0].get('source', [''])[:3]), language='python')
                    except:
                        st.text_area("RAW NOTEBOOK CONTENT (FIRST 1000 CHARS)", content[:1000])
                        
                except Exception as e:
                    st.error(f"ERROR reading notebook: {e}")
        
        # Execute if status is Running
        if st.session_state.execution_status.get(selected_model) == "Running":
            python_path = os.path.join(selected_model, "main_converted.py")
            
            with st.spinner(f"EXECUTING {selected_model} OPTIMIZATION..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "main_converted.py"],
                        cwd=selected_model,
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout
                    )
                    
                    if result.returncode == 0:
                        st.session_state.execution_status[selected_model] = "Completed"
                        st.success(f"✅ {selected_model} OPTIMIZATION COMPLETED!")
                        st.text_area("EXECUTION OUTPUT", result.stdout, height=200)
                    else:
                        st.session_state.execution_status[selected_model] = "Error"
                        st.error(f"❌ {selected_model} OPTIMIZATION FAILED!")
                        st.text_area("ERROR LOG", result.stderr, height=200)
                
                except subprocess.TimeoutExpired:
                    st.session_state.execution_status[selected_model] = "Error"
                    st.error("❌ EXECUTION TIMEOUT!")
                except Exception as e:
                    st.session_state.execution_status[selected_model] = "Error"
                    st.error(f"❌ EXECUTION ERROR: {e}")
                
                st.rerun()
    else:
        st.warning("❌ main.ipynb NOT FOUND IN SYSTEM")

# Footer with hacker styling
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #00ff00; font-family: 'Courier New', monospace;">
    <p>🚀 DENSENET OPTIMIZATION NEURAL NETWORK DASHBOARD | POWERED BY STREAMLIT</p>
    <p>📊 ANALYZE HYPERPARAMETER OPTIMIZATION RESULTS AND VISUALIZATIONS</p>
    <p style="font-size: 0.8em; color: #00aa00;">SYSTEM STATUS: ONLINE | READY FOR NEURAL NETWORK ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

# Sidebar file browser with hacker styling
st.sidebar.markdown("---")
st.sidebar.subheader(f"📁 {selected_model} FILE SYSTEM")

if model_files:
    for filename in sorted(model_files.keys()):
        file_icon = "📊" if filename.endswith('.csv') else "🖼️" if filename.endswith('.png') else "📝" if filename.endswith(('.txt', '.md')) else "📓" if filename.endswith('.ipynb') else "📄"
        st.sidebar.text(f"{file_icon} {filename}")
else:
    st.sidebar.info("NO FILES DETECTED")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ SYSTEM INFO")
st.sidebar.info("""
NEURAL NETWORK ANALYSIS DASHBOARD

FUNCTIONS:
- 🏆 View optimal hyperparameters
- 📊 Analyze performance metrics  
- 📈 Display neural network visualizations
- 📋 Browse system data files
- 🔧 Manage Jupyter notebook execution

SELECT DIFFERENT MODELS TO COMPARE RESULTS

SYSTEM STATUS: ONLINE
""")
