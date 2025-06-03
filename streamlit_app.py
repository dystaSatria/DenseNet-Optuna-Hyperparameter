import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import json
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import tempfile

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
    .notebook-cell {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
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
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "main.ipynb")):
        available_dirs.append(model_dir)

if not available_dirs:
    st.error("❌ No model directories with main.ipynb found!")
    st.info("Make sure you have folders: DenseNet121, DenseNet169, DenseNet201 with main.ipynb files")
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
    ["Execute Notebooks", "Convert to Python", "Display Only"],
    help="Execute: Run notebook directly\nConvert: Convert to .py then run\nDisplay: Show interface only"
)

# Execution options
execution_method = st.sidebar.selectbox(
    "Execution Method:",
    ["nbclient (Recommended)", "jupyter nbconvert", "papermill"],
    help="Choose how to execute the notebooks"
)

# Advanced options
st.sidebar.subheader("🔧 Advanced Settings")
show_output = st.sidebar.checkbox("Show cell outputs", value=True)
save_executed_nb = st.sidebar.checkbox("Save executed notebooks", value=True)
timeout_minutes = st.sidebar.number_input("Timeout (minutes)", min_value=5, max_value=180, value=60)

# Initialize session state
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = {}
if 'notebook_outputs' not in st.session_state:
    st.session_state.notebook_outputs = {}

# Main content
st.header("🚀 Notebook Execution Dashboard")

# Function to execute notebook using nbclient
def execute_notebook_nbclient(notebook_path, model_dir):
    """Execute notebook using nbclient"""
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create client
        client = NotebookClient(
            nb, 
            timeout=timeout_minutes * 60,
            kernel_name='python3'
        )
        
        # Execute in the model directory
        original_dir = os.getcwd()
        os.chdir(model_dir)
        
        # Execute notebook
        client.execute()
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Save executed notebook if requested
        if save_executed_nb:
            output_path = os.path.join(model_dir, "main_executed.ipynb")
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        
        # Extract outputs
        outputs = []
        for cell in nb.cells:
            if cell.cell_type == 'code' and cell.outputs:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        outputs.append(output.text)
                    elif output.output_type == 'execute_result':
                        if 'text/plain' in output.data:
                            outputs.append(output.data['text/plain'])
        
        return True, "\n".join(outputs), ""
        
    except CellExecutionError as e:
        os.chdir(original_dir)
        return False, "", f"Cell execution error: {str(e)}"
    except Exception as e:
        os.chdir(original_dir)
        return False, "", f"Execution error: {str(e)}"

# Function to execute using jupyter nbconvert
def execute_notebook_nbconvert(notebook_path, model_dir):
    """Execute notebook using jupyter nbconvert"""
    try:
        original_dir = os.getcwd()
        os.chdir(model_dir)
        
        result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'notebook',
            '--execute', 
            '--inplace' if not save_executed_nb else '--output', 'main_executed.ipynb',
            'main.ipynb'
        ], capture_output=True, text=True, timeout=timeout_minutes * 60)
        
        os.chdir(original_dir)
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        return False, "", f"Execution timed out ({timeout_minutes} minutes)"
    except Exception as e:
        os.chdir(original_dir)
        return False, "", str(e)

# Function to convert notebook to python and execute
def convert_and_execute_notebook(notebook_path, model_dir):
    """Convert notebook to Python script and execute"""
    try:
        original_dir = os.getcwd()
        model_path = os.path.join(original_dir, model_dir)
        
        # Convert to Python
        convert_result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'python',
            '--output', 'main_converted.py',
            notebook_path
        ], capture_output=True, text=True)
        
        if convert_result.returncode != 0:
            return False, "", f"Conversion failed: {convert_result.stderr}"
        
        # Execute Python script
        python_file = os.path.join(model_path, "main_converted.py")
        
        result = subprocess.run(
            [sys.executable, python_file],
            cwd=model_path,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60
        )
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out ({timeout_minutes} minutes)"
    except Exception as e:
        return False, "", str(e)

# Function to get notebook preview
def get_notebook_preview(notebook_path, max_cells=3):
    """Get preview of notebook cells"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        preview_cells = []
        code_cell_count = 0
        
        for cell in nb.cells:
            if cell.cell_type == 'code' and code_cell_count < max_cells:
                preview_cells.append({
                    'type': 'code',
                    'source': cell.source[:200] + "..." if len(cell.source) > 200 else cell.source
                })
                code_cell_count += 1
            elif cell.cell_type == 'markdown' and len(preview_cells) < max_cells:
                preview_cells.append({
                    'type': 'markdown',
                    'source': cell.source[:100] + "..." if len(cell.source) > 100 else cell.source
                })
        
        return preview_cells
    except:
        return []

# Function to check if results exist
def check_results_exist(model_dir):
    """Check if optimization results exist"""
    results_files = [
        os.path.join(model_dir, "best_params.json"),
        os.path.join(model_dir, "optimization_history.csv"),
        os.path.join(model_dir, "study.pkl"),
        os.path.join(model_dir, "main_executed.ipynb")
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
            <p><strong>File:</strong> 📓 main.ipynb</p>
        </div>
        """, unsafe_allow_html=True)

# Individual model controls
st.header("🎮 Individual Notebook Controls")

for model_dir in selected_models:
    with st.expander(f"📓 {model_dir} Notebook Controls", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**Directory:** `{model_dir}/`")
            st.write(f"**Notebook:** `{model_dir}/main.ipynb`")
            
            # Check if main.ipynb exists
            notebook_path = os.path.join(model_dir, "main.ipynb")
            if os.path.exists(notebook_path):
                st.success("✅ main.ipynb found")
                
                # Show notebook preview
                preview_cells = get_notebook_preview(notebook_path)
                if preview_cells:
                    st.write("**Notebook Preview:**")
                    for i, cell in enumerate(preview_cells):
                        if cell['type'] == 'code':
                            st.code(cell['source'], language='python')
                        else:
                            st.markdown(f"*Markdown:* {cell['source']}")
                        if i < len(preview_cells) - 1:
                            st.markdown("---")
            else:
                st.error("❌ main.ipynb not found")
        
        with col2:
            # Run button
            if st.button(f"🚀 Run {model_dir}", key=f"run_{model_dir}"):
                if run_mode != "Display Only":
                    st.session_state.execution_status[model_dir] = "Running"
                    st.rerun()
                else:
                    st.info("Display Only mode - execution disabled")
            
            # View results button
            if check_results_exist(model_dir):
                if st.button(f"📊 View Results", key=f"results_{model_dir}"):
                    st.session_state[f"show_results_{model_dir}"] = True
            
            # View notebook button
            if st.button(f"📓 View Notebook", key=f"view_nb_{model_dir}"):
                st.session_state[f"show_notebook_{model_dir}"] = True
        
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
            notebook_path = os.path.join(model_dir, "main.ipynb")
            
            with st.spinner(f"Executing {model_dir} notebook..."):
                # Choose execution method
                if execution_method == "nbclient (Recommended)":
                    success, stdout, stderr = execute_notebook_nbclient(notebook_path, model_dir)
                elif execution_method == "jupyter nbconvert":
                    success, stdout, stderr = execute_notebook_nbconvert(notebook_path, model_dir)
                else:  # Convert to Python
                    success, stdout, stderr = convert_and_execute_notebook(notebook_path, model_dir)
                
                if success:
                    st.session_state.execution_status[model_dir] = "Completed"
                    st.success(f"✅ {model_dir} notebook executed successfully!")
                else:
                    st.session_state.execution_status[model_dir] = "Error"
                    st.error(f"❌ {model_dir} notebook execution failed!")
                
                # Store logs and outputs
                st.session_state.execution_logs[model_dir] = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": execution_method
                }
                
                if show_output and (stdout or stderr):
                    st.subheader(f"📝 Execution Output for {model_dir}")
                    if stdout:
                        st.text_area("Output:", stdout, height=200)
                    if stderr:
                        st.text_area("Errors:", stderr, height=100)
                
                st.rerun()
        
        # Show notebook content if requested
        if st.session_state.get(f"show_notebook_{model_dir}", False):
            st.subheader(f"📓 Notebook Content: {model_dir}")
            
            notebook_path = os.path.join(model_dir, "main.ipynb")
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                for i, cell in enumerate(nb.cells):
                    st.markdown(f"**Cell {i+1} ({cell.cell_type}):**")
                    
                    if cell.cell_type == 'code':
                        st.code(cell.source, language='python')
                        if cell.outputs and show_output:
                            st.markdown("*Output:*")
                            for output in cell.outputs:
                                if output.output_type == 'stream':
                                    st.text(output.text)
                                elif output.output_type == 'execute_result':
                                    if 'text/plain' in output.data:
                                        st.text(output.data['text/plain'])
                    else:
                        st.markdown(cell.source)
                    
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Could not read notebook: {e}")
            
            if st.button(f"❌ Close Notebook View", key=f"close_nb_{model_dir}"):
                st.session_state[f"show_notebook_{model_dir}"] = False
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
            
            # Check for executed notebook
            executed_nb = os.path.join(model_dir, "main_executed.ipynb")
            if os.path.exists(executed_nb):
                st.info("✅ Executed notebook saved as main_executed.ipynb")
            
            if st.button(f"❌ Close Results", key=f"close_results_{model_dir}"):
                st.session_state[f"show_results_{model_dir}"] = False
                st.rerun()

# Batch operations
st.header("🔄 Batch Operations")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 Run All Selected"):
        if run_mode != "Display Only":
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
                file_name=f"notebook_execution_logs_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

with col4:
    if st.button("🧹 Clear All Logs"):
        st.session_state.execution_logs = {}
        st.session_state.notebook_outputs = {}
        st.rerun()

# Execution logs
if st.session_state.execution_logs:
    st.header("📝 Execution Logs")
    
    for model, logs in st.session_state.execution_logs.items():
        with st.expander(f"📋 {model} Logs - {logs['timestamp']} ({logs.get('method', 'Unknown')})"):
            if logs['stdout']:
                st.subheader("Output")
                st.code(logs['stdout'])
            if logs['stderr']:
                st.subheader("Errors")
                st.code(logs['stderr'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 DenseNet Notebook Execution Runner | Built with Streamlit</p>
    <p>📓 Execute Jupyter notebooks for hyperparameter optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
This app executes Jupyter notebooks for DenseNet hyperparameter optimization.

**Requirements:**
- Each model folder must contain main.ipynb
- nbclient, nbformat packages installed
- Jupyter environment properly configured

**Execution Methods:**
- **nbclient**: Direct notebook execution (recommended)
- **jupyter nbconvert**: Convert and execute
- **Convert to Python**: Convert to .py then run
""")

st.sidebar.subheader("📂 Directory Structure")
st.sidebar.code("""
DenseNet-Optuna-Hyperparameter/
├── streamlit_app.py
├── requirements.txt
├── DenseNet121/
│   └── main.ipynb
├── DenseNet169/
│   └── main.ipynb
└── DenseNet201/
    └── main.ipynb
""")

st.sidebar.subheader("⚠️ Notes")
st.sidebar.warning("""
- Notebook execution may take considerable time
- Ensure sufficient computational resources
- Large outputs may affect performance
- Save important results before re-running
""")
