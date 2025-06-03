import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import json
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="DenseNet Hyperparameter Optimization",
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
    .run-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .folder-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .status-running {
        color: #ffa500;
        font-weight: bold;
    }
    .status-completed {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 DenseNet Hyperparameter Optimization Runner</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("---")

# Model directories
model_dirs = ["DenseNet121", "DenseNet169", "DenseNet201"]
available_dirs = []

# Check which directories exist
for model_dir in model_dirs:
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "main.py")):
        available_dirs.append(model_dir)

if not available_dirs:
    st.error("❌ No model directories with main.py found!")
    st.info("Make sure you have folders: DenseNet121, DenseNet169, DenseNet201 with main.py files")
    st.stop()

# Sidebar options
st.sidebar.subheader("🎯 Model Selection")
selected_models = st.sidebar.multiselect(
    "Select models to run:",
    available_dirs,
    default=available_dirs,
    help="Choose which DenseNet models to optimize"
)

# Run options
st.sidebar.subheader("⚙️ Run Options")
run_mode = st.sidebar.radio(
    "Execution Mode:",
    ["Sequential", "Display Only"],
    help="Sequential: Run models one by one\nDisplay Only: Show interface without running"
)

# Advanced options
st.sidebar.subheader("🔧 Advanced Settings")
show_output = st.sidebar.checkbox("Show real-time output", value=True)
save_logs = st.sidebar.checkbox("Save execution logs", value=True)

# Initialize session state
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = {}

# Main content
st.header("🚀 Model Execution Dashboard")

# Function to run main.py in a specific directory
def run_model_optimization(model_dir):
    """Run main.py in the specified model directory"""
    try:
        # Change to model directory
        original_dir = os.getcwd()
        model_path = os.path.join(original_dir, model_dir)
        
        if not os.path.exists(os.path.join(model_path, "main.py")):
            return False, f"main.py not found in {model_dir}"
        
        # Run the main.py script
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=model_path,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out (1 hour limit)"
    except Exception as e:
        return False, "", str(e)

# Function to check if results exist
def check_results_exist(model_dir):
    """Check if optimization results exist"""
    results_files = [
        os.path.join(model_dir, "best_params.json"),
        os.path.join(model_dir, "optimization_history.csv"),
        os.path.join(model_dir, "study.pkl")
    ]
    return any(os.path.exists(f) for f in results_files)

# Model status overview
st.subheader("📊 Model Status Overview")

cols = st.columns(len(available_dirs))
for i, model_dir in enumerate(available_dirs):
    with cols[i]:
        has_results = check_results_exist(model_dir)
        status = st.session_state.execution_status.get(model_dir, "Not Started")
        
        if status == "Running":
            status_color = "🟡"
            status_text = "status-running"
        elif status == "Completed":
            status_color = "🟢"
            status_text = "status-completed"
        elif status == "Error":
            status_color = "🔴"
            status_text = "status-error"
        else:
            status_color = "⚪"
            status_text = ""
        
        st.markdown(f"""
        <div class="folder-card">
            <h4>{status_color} {model_dir}</h4>
            <p><strong>Status:</strong> <span class="{status_text}">{status}</span></p>
            <p><strong>Results:</strong> {"✅ Available" if has_results else "❌ Not Available"}</p>
        </div>
        """, unsafe_allow_html=True)

# Individual model controls
st.header("🎮 Individual Model Controls")

for model_dir in selected_models:
    with st.expander(f"🔧 {model_dir} Controls", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**Directory:** `{model_dir}/`")
            st.write(f"**Main Script:** `{model_dir}/main.py`")
            
            # Check if main.py exists
            main_py_path = os.path.join(model_dir, "main.py")
            if os.path.exists(main_py_path):
                st.success("✅ main.py found")
                
                # Try to read first few lines to show preview
                try:
                    with open(main_py_path, 'r', encoding='utf-8') as f:
                        preview = f.readlines()[:5]
                    st.code("".join(preview), language="python")
                except:
                    st.info("Could not preview file content")
            else:
                st.error("❌ main.py not found")
        
        with col2:
            # Run button
            if st.button(f"🚀 Run {model_dir}", key=f"run_{model_dir}"):
                if run_mode == "Sequential":
                    st.session_state.execution_status[model_dir] = "Running"
                    st.rerun()
                else:
                    st.info("Display Only mode - execution disabled")
            
            # View results button
            if check_results_exist(model_dir):
                if st.button(f"📊 View Results", key=f"results_{model_dir}"):
                    st.session_state[f"show_results_{model_dir}"] = True
        
        with col3:
            # Status indicator
            status = st.session_state.execution_status.get(model_dir, "Not Started")
            if status == "Running":
                st.markdown("🟡 **Running...**")
            elif status == "Completed":
                st.markdown("🟢 **Completed**")
            elif status == "Error":
                st.markdown("🔴 **Error**")
            else:
                st.markdown("⚪ **Ready**")
        
        # Execute if status is Running
        if st.session_state.execution_status.get(model_dir) == "Running":
            with st.spinner(f"Running optimization for {model_dir}..."):
                success, stdout, stderr = run_model_optimization(model_dir)
                
                if success:
                    st.session_state.execution_status[model_dir] = "Completed"
                    st.success(f"✅ {model_dir} optimization completed successfully!")
                else:
                    st.session_state.execution_status[model_dir] = "Error"
                    st.error(f"❌ {model_dir} optimization failed!")
                
                # Store logs
                st.session_state.execution_logs[model_dir] = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if show_output and (stdout or stderr):
                    st.subheader(f"📝 Execution Output for {model_dir}")
                    if stdout:
                        st.text_area("Standard Output:", stdout, height=200)
                    if stderr:
                        st.text_area("Standard Error:", stderr, height=100)
                
                st.rerun()
        
        # Show results if requested
        if st.session_state.get(f"show_results_{model_dir}", False):
            st.subheader(f"📊 Results for {model_dir}")
            
            # Try to load and display results
            results_file = os.path.join(model_dir, "best_params.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        best_params = json.load(f)
                    
                    st.json(best_params)
                except:
                    st.error("Could not load results file")
            
            # Try to load optimization history
            history_file = os.path.join(model_dir, "optimization_history.csv")
            if os.path.exists(history_file):
                try:
                    history_df = pd.read_csv(history_file)
                    st.dataframe(history_df.head(10))
                    
                    # Simple plot
                    if 'value' in history_df.columns:
                        st.line_chart(history_df['value'])
                except:
                    st.error("Could not load history file")
            
            if st.button(f"❌ Close Results", key=f"close_{model_dir}"):
                st.session_state[f"show_results_{model_dir}"] = False
                st.rerun()

# Batch operations
st.header("🔄 Batch Operations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Run All Selected Models"):
        if run_mode == "Sequential":
            for model in selected_models:
                st.session_state.execution_status[model] = "Running"
            st.rerun()
        else:
            st.info("Display Only mode - execution disabled")

with col2:
    if st.button("🔄 Reset All Status"):
        for model in available_dirs:
            if model in st.session_state.execution_status:
                del st.session_state.execution_status[model]
        st.rerun()

with col3:
    if st.button("📋 Export Logs"):
        if st.session_state.execution_logs:
            logs_json = json.dumps(st.session_state.execution_logs, indent=2)
            st.download_button(
                label="Download Logs",
                data=logs_json,
                file_name=f"optimization_logs_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Execution logs
if st.session_state.execution_logs:
    st.header("📝 Execution Logs")
    
    for model, logs in st.session_state.execution_logs.items():
        with st.expander(f"📋 {model} Logs - {logs['timestamp']}"):
            if logs['stdout']:
                st.subheader("Standard Output")
                st.code(logs['stdout'])
            if logs['stderr']:
                st.subheader("Standard Error")
                st.code(logs['stderr'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 DenseNet Hyperparameter Optimization Runner | Built with Streamlit</p>
    <p>📁 Make sure each model directory contains a main.py file</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
This app allows you to run hyperparameter optimization scripts for different DenseNet models.

**Requirements:**
- Each model folder must contain main.py
- Python environment with required packages
- Sufficient computational resources

**Features:**
- Run individual or batch optimizations
- Real-time status monitoring
- Execution logs and output display
- Results viewing and export
""")

st.sidebar.subheader("📂 Directory Structure")
st.sidebar.code("""
DenseNet-Optuna-Hyperparameter/
├── streamlit_app.py
├── requirements.txt
├── DenseNet121/
│   └── main.py
├── DenseNet169/
│   └── main.py
└── DenseNet201/
    └── main.py
""")
