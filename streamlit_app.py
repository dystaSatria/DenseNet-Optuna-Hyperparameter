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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .folder-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .status-completed {
        color: #28a745;
        font-weight: bold;
    }
    .status-running {
        color: #ffa500;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .results-section {
        background-color: #fff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Alzheimer Classification | DenseNet Hyperparameter Optimization Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("---")

# Model directories
model_dirs = ["DenseNet121", "DenseNet169", "DenseNet201"]
available_dirs = []

# Check which directories exist
for model_dir in model_dirs:
    if os.path.exists(model_dir):
        available_dirs.append(model_dir)

if not available_dirs:
    st.error("❌ No model directories found!")
    st.info("Make sure you have folders: DenseNet121, DenseNet169, DenseNet201")
    st.stop()

# Sidebar options
st.sidebar.subheader("🎯 Model Selection")
selected_model = st.sidebar.selectbox(
    "Select model to analyze:",
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
        st.error(f"Error loading {file_path}: {e}")
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
        st.error(f"Error loading image {image_path}: {e}")
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
st.header(f"🚀 {selected_model} Analysis Dashboard")

# Get files in selected directory
model_files = get_directory_files(selected_model)

# Display overview
col1, col2, col3 = st.columns(3)

with col1:
    csv_files = [f for f in model_files.keys() if f.endswith('.csv')]
    st.metric("📊 CSV Files", len(csv_files))

with col2:
    png_files = [f for f in model_files.keys() if f.endswith('.png')]
    st.metric("📈 Visualizations", len(png_files))

with col3:
    has_notebook = 'main.ipynb' in model_files
    st.metric("📓 Notebook", "✅" if has_notebook else "❌")

# Status card
st.markdown(f"""
<div class="folder-card">
    <h4>📁 {selected_model} Status</h4>
    <p><strong>Directory:</strong> {selected_model}/</p>
    <p><strong>Total Files:</strong> {len(model_files)}</p>
    <p><strong>Has Results:</strong> {"✅ Yes" if len(csv_files) > 0 else "❌ No"}</p>
    <p><strong>Has Visualizations:</strong> {"✅ Yes" if len(png_files) > 0 else "❌ No"}</p>
</div>
""", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Best Results", "📊 Metrics Analysis", "📈 Visualizations", "📋 All Data", "🔧 Notebook Management"])

with tab1:
    st.header("🏆 Best Hyperparameters & Results")
    
    # Load best hyperparameters
    best_params = read_best_hyperparameters(selected_model)
    if best_params:
        st.subheader("📝 Best Hyperparameters")
        st.text_area("Hyperparameters:", best_params, height=200)
    else:
        st.info("No best_hyperparameters.txt file found")
    
    # Show key metrics files
    col1, col2 = st.columns(2)
    
    with col1:
        # Best hybrid metrics
        best_hybrid_file = os.path.join(selected_model, "best_hybrid_per_class_metrics.csv")
        if os.path.exists(best_hybrid_file):
            st.subheader("🎯 Best Hybrid Model Metrics")
            df_hybrid = load_csv_data(best_hybrid_file)
            if df_hybrid is not None:
                st.dataframe(df_hybrid, use_container_width=True)
    
    with col2:
        # DenseNet metrics
        densenet_metrics_file = os.path.join(selected_model, "densenet_per_class_metrics.csv")
        if os.path.exists(densenet_metrics_file):
            st.subheader("🧠 DenseNet Model Metrics")
            df_densenet = load_csv_data(densenet_metrics_file)
            if df_densenet is not None:
                st.dataframe(df_densenet, use_container_width=True)
    
    # ROC AUC Comparison
    roc_file = os.path.join(selected_model, "roc_auc_comparison.csv")
    if os.path.exists(roc_file):
        st.subheader("📈 ROC AUC Comparison")
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
    st.header("📊 Detailed Metrics Analysis")
    
    # All models metrics
    all_models_file = os.path.join(selected_model, "all_models_metrics.csv")
    if os.path.exists(all_models_file):
        st.subheader("🔍 All Models Comparison")
        df_all = load_csv_data(all_models_file)
        if df_all is not None:
            st.dataframe(df_all, use_container_width=True)
            
            # Show summary statistics
            if not df_all.empty:
                st.subheader("📈 Summary Statistics")
                numeric_cols = df_all.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df_all[numeric_cols].describe(), use_container_width=True)
    
    # SVM Optimization Results
    svm_file = os.path.join(selected_model, "svm_optimization_results.csv")
    if os.path.exists(svm_file):
        st.subheader("⚙️ SVM Optimization Results")
        df_svm = load_csv_data(svm_file)
        if df_svm is not None:
            st.dataframe(df_svm.head(20), use_container_width=True)
            
            # Show best SVM results
            if 'score' in df_svm.columns or 'accuracy' in df_svm.columns:
                score_col = 'score' if 'score' in df_svm.columns else 'accuracy'
                best_svm = df_svm.loc[df_svm[score_col].idxmax()]
                st.subheader("🏅 Best SVM Configuration")
                st.json(best_svm.to_dict())

with tab3:
    st.header("📈 Visualizations")
    
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
            st.subheader("🎯 Confusion Matrices")
            cols = st.columns(min(2, len(confusion_matrices)))
            for i, plot_file in enumerate(confusion_matrices):
                with cols[i % 2]:
                    display_image(model_files[plot_file], plot_file.replace('.png', '').replace('_', ' ').title())
        
        if training_plots:
            st.subheader("📚 Training History")
            for plot_file in training_plots:
                display_image(model_files[plot_file], plot_file.replace('.png', '').replace('_', ' ').title())
        
        if comparison_plots:
            st.subheader("⚖️ Model Comparisons")
            cols = st.columns(min(2, len(comparison_plots)))
            for i, plot_file in enumerate(comparison_plots):
                with cols[i % 2]:
                    display_image(model_files[plot_file], plot_file.replace('.png', '').replace('_', ' ').title())
        
        if roc_plots:
            st.subheader("📈 ROC Curves")
            for plot_file in roc_plots:
                display_image(model_files[plot_file], plot_file.replace('.png', '').replace('_', ' ').title())
        
        if other_plots:
            st.subheader("📊 Other Visualizations")
            cols = st.columns(min(2, len(other_plots)))
            for i, plot_file in enumerate(other_plots):
                with cols[i % 2]:
                    display_image(model_files[plot_file], plot_file.replace('.png', '').replace('_', ' ').title())
    else:
        st.info("No visualization files (.png) found in this directory")

with tab4:
    st.header("📋 All Files & Data")
    
    # Show all files in directory
    st.subheader(f"📁 Files in {selected_model}/")
    
    file_types = {}
    for filename, filepath in model_files.items():
        ext = filename.split('.')[-1] if '.' in filename else 'no extension'
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(filename)
    
    for ext, files in file_types.items():
        with st.expander(f"📄 {ext.upper()} Files ({len(files)})"):
            for filename in files:
                filepath = model_files[filename]
                st.write(f"**{filename}**")
                
                if filename.endswith('.csv'):
                    if st.button(f"📊 View {filename}", key=f"view_{filename}"):
                        df = load_csv_data(filepath)
                        if df is not None:
                            st.dataframe(df, use_container_width=True)
                
                elif filename.endswith('.txt') or filename.endswith('.md'):
                    if st.button(f"📝 View {filename}", key=f"view_{filename}"):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            st.text_area(f"Content of {filename}:", content, height=300)
                        except Exception as e:
                            st.error(f"Error reading {filename}: {e}")
                
                elif filename.endswith('.png'):
                    if st.button(f"🖼️ View {filename}", key=f"view_{filename}"):
                        display_image(filepath, filename)

with tab5:
    st.header("🔧 Notebook Management")
    
    # Check if main.ipynb exists
    notebook_path = os.path.join(selected_model, "main.ipynb")
    
    if os.path.exists(notebook_path):
        st.success("✅ main.ipynb found")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Convert to Python"):
                python_path = os.path.join(selected_model, "main_converted.py")
                
                with st.spinner("Converting notebook..."):
                    success, result = convert_notebook_to_python(notebook_path, python_path)
                    
                    if success:
                        st.success("✅ Converted successfully!")
                        st.text_area("Converted Python Code Preview:", result[:500] + "..." if len(result) > 500 else result, height=200)
                    else:
                        st.error(f"❌ Conversion failed: {result}")
        
        with col2:
            if st.button("🚀 Run Optimization"):
                python_path = os.path.join(selected_model, "main_converted.py")
                if os.path.exists(python_path):
                    st.session_state.execution_status[selected_model] = "Running"
                    st.rerun()
                else:
                    st.error("❌ Please convert notebook to Python first!")
        
        with col3:
            if st.button("📓 View Notebook Content"):
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Try to parse notebook JSON
                    try:
                        nb_data = json.loads(content)
                        if 'cells' in nb_data:
                            st.write(f"📊 Notebook has {len(nb_data['cells'])} cells")
                            
                            # Show first few code cells
                            code_cells = [cell for cell in nb_data['cells'] if cell.get('cell_type') == 'code']
                            if code_cells:
                                st.write(f"🔍 Found {len(code_cells)} code cells")
                                st.code(str(code_cells[0].get('source', [''])[:3]), language='python')
                    except:
                        st.text_area("Raw notebook content (first 1000 chars):", content[:1000])
                        
                except Exception as e:
                    st.error(f"Error reading notebook: {e}")
        
        # Execute if status is Running
        if st.session_state.execution_status.get(selected_model) == "Running":
            python_path = os.path.join(selected_model, "main_converted.py")
            
            with st.spinner(f"Executing {selected_model} optimization..."):
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
                        st.success(f"✅ {selected_model} optimization completed!")
                        st.text_area("Output:", result.stdout, height=200)
                    else:
                        st.session_state.execution_status[selected_model] = "Error"
                        st.error(f"❌ {selected_model} optimization failed!")
                        st.text_area("Error:", result.stderr, height=200)
                
                except subprocess.TimeoutExpired:
                    st.session_state.execution_status[selected_model] = "Error"
                    st.error("❌ Execution timed out!")
                except Exception as e:
                    st.session_state.execution_status[selected_model] = "Error"
                    st.error(f"❌ Execution error: {e}")
                
                st.rerun()
    else:
        st.warning("❌ main.ipynb not found in this directory")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 DenseNet Optimization Results Dashboard | Built with Streamlit</p>
    <p>📊 Analyze hyperparameter optimization results and visualizations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar file browser
st.sidebar.markdown("---")
st.sidebar.subheader(f"📁 {selected_model} Files")

if model_files:
    for filename in sorted(model_files.keys()):
        file_icon = "📊" if filename.endswith('.csv') else "🖼️" if filename.endswith('.png') else "📝" if filename.endswith(('.txt', '.md')) else "📓" if filename.endswith('.ipynb') else "📄"
        st.sidebar.text(f"{file_icon} {filename}")
else:
    st.sidebar.info("No files found")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
This dashboard analyzes existing optimization results and allows you to:

- 🏆 View best hyperparameters
- 📊 Analyze performance metrics  
- 📈 Display visualizations
- 📋 Browse all data files
- 🔧 Manage Jupyter notebooks

Select different models from the dropdown to compare results.
""")
