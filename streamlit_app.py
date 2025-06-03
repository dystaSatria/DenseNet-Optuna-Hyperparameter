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
    .notebook-preview {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
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
    
    # Show current directory structure for debugging
    st.subheader("📁 Current Directory Structure")
    try:
        current_files = os.listdir(".")
        st.write("**Files in current directory:**", current_files)
        
        for item in current_files:
            if os.path.isdir(item):
                try:
                    sub_files = os.listdir(item)
                    st.write(f"**Files in {item}/:**", sub_files)
                except:
                    st.write(f"**{item}/:** Cannot access")
    except Exception as e:
        st.error(f"Cannot read directory: {e}")
    
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
    ["Convert & Execute", "Display Only"],
    help="Convert: Convert .ipynb to .py then run\nDisplay: Show interface only"
)

# Advanced options
st.sidebar.subheader("🔧 Advanced Settings")
show_output = st.sidebar.checkbox("Show execution output", value=True)
timeout_minutes = st.sidebar.number_input("Timeout (minutes)", min_value=5, max_value=180, value=60)

# Initialize session state
if 'execution_status' not in st.session_state:
    st.session_state.execution_status = {}
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = {}

# Main content
st.header("🚀 Notebook Execution Dashboard")

# Function to read notebook file and show preview
def get_notebook_preview(notebook_path, max_lines=10):
    """Get preview of notebook file"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as JSON to extract code cells
        try:
            import json
            nb_data = json.loads(content)
            
            if 'cells' in nb_data:
                code_cells = []
                for cell in nb_data['cells']:
                    if cell.get('cell_type') == 'code' and cell.get('source'):
                        source = cell['source']
                        if isinstance(source, list):
                            source = ''.join(source)
                        code_cells.append(source)
                
                if code_cells:
                    preview = code_cells[0][:500] + "..." if len(code_cells[0]) > 500 else code_cells[0]
                    return preview, len(code_cells)
        except:
            pass
        
        # Fallback: show raw content preview
        lines = content.split('\n')
        preview_lines = lines[:max_lines]
        return '\n'.join(preview_lines) + f"\n... ({len(lines)} total lines)", 0
        
    except Exception as e:
        return f"Error reading file: {e}", 0

# Function to convert notebook to python
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

# Function to execute Python script
def execute_python_script(script_path, working_dir):
    """Execute Python script in specified directory"""
    try:
        original_dir = os.getcwd()
        os.chdir(working_dir)
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60
        )
        
        os.chdir(original_dir)
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        return False, "", f"Execution timed out ({timeout_minutes} minutes)"
    except Exception as e:
        os.chdir(original_dir)
        return False, "", str(e)

# Function to check if results exist
def check_results_exist(model_dir):
    """Check if optimization results exist"""
    results_files = [
        os.path.join(model_dir, "best_params.json"),
        os.path.join(model_dir, "optimization_history.csv"),
        os.path.join(model_dir, "study.pkl"),
        os.path.join(model_dir, "main_converted.py")
    ]
    existing_files = [f for f in results_files if os.path.exists(f)]
    return len(existing_files) > 0, existing_files

# Model status overview
st.subheader("📊 Model Status Overview")

cols = st.columns(len(available_dirs))
for i, model_dir in enumerate(available_dirs):
    with cols[i]:
        has_results, result_files = check_results_exist(model_dir)
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
st.header("🎮 Individual Model Controls")

for model_dir in selected_models:
    with st.expander(f"📓 {model_dir} Controls", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**Directory:** `{model_dir}/`")
            st.write(f"**Notebook:** `{model_dir}/main.ipynb`")
            
            # Check if main.ipynb exists and show preview
            notebook_path = os.path.join(model_dir, "main.ipynb")
            if os.path.exists(notebook_path):
                st.success("✅ main.ipynb found")
                
                # Show notebook preview
                preview, num_cells = get_notebook_preview(notebook_path)
                if preview:
                    st.write("**Notebook Preview:**")
                    st.markdown(f"""
                    <div class="notebook-preview">
                    {preview}
                    </div>
                    """, unsafe_allow_html=True)
                    if num_cells > 0:
                        st.info(f"📊 Found {num_cells} code cells in notebook")
            else:
                st.error("❌ main.ipynb not found")
        
        with col2:
            # Convert button
            if st.button(f"🔄 Convert to Python", key=f"convert_{model_dir}"):
                notebook_path = os.path.join(model_dir, "main.ipynb")
                python_path = os.path.join(model_dir, "main_converted.py")
                
                with st.spinner("Converting notebook..."):
                    success, result = convert_notebook_to_python(notebook_path, python_path)
                    
                    if success:
                        st.success("✅ Converted successfully!")
                        st.code(result[:300] + "..." if len(result) > 300 else result, language="python")
                    else:
                        st.error(f"❌ Conversion failed: {result}")
            
            # Run button
            if st.button(f"🚀 Run {model_dir}", key=f"run_{model_dir}"):
                if run_mode != "Display Only":
                    # Check if converted Python file exists
                    python_path = os.path.join(model_dir, "main_converted.py")
                    if os.path.exists(python_path):
                        st.session_state.execution_status[model_dir] = "Running"
                        st.rerun()
                    else:
                        st.error("❌ Please convert notebook to Python first!")
                else:
                    st.info("Display Only mode - execution disabled")
        
        with col3:
            # Status and results
            status = st.session_state.execution_status.get(model_dir, "Not Started")
            if status == "Running":
                st.markdown("🟡 **Running...**")
            elif status == "Completed":
                st.markdown("🟢 **Completed**")
            elif status == "Error":
                st.markdown("🔴 **Error**")
            else:
                st.markdown("⚪ **Ready**")
            
            # View results button
            has_results, result_files = check_results_exist(model_dir)
            if has_results:
                if st.button(f"📊 View Results", key=f"results_{model_dir}"):
                    st.session_state[f"show_results_{model_dir}"] = True
        
        # Execute if status is Running
        if st.session_state.execution_status.get(model_dir) == "Running":
            python_path = os.path.join(model_dir, "main_converted.py")
            
            with st.spinner(f"Executing {model_dir} optimization..."):
                success, stdout, stderr = execute_python_script("main_converted.py", model_dir)
                
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
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": "converted_python"
                }
                
                if show_output and (stdout or stderr):
                    st.subheader(f"📝 Execution Output for {model_dir}")
                    if stdout:
                        st.text_area("Output:", stdout, height=200, key=f"stdout_{model_dir}")
                    if stderr:
                        st.text_area("Errors:", stderr, height=100, key=f"stderr_{model_dir}")
                
                st.rerun()
        
        # Show results if requested
        if st.session_state.get(f"show_results_{model_dir}", False):
            st.subheader(f"📊 Results for {model_dir}")
            
            has_results, result_files = check_results_exist(model_dir)
            
            if result_files:
                st.write("**Available result files:**")
                for file_path in result_files:
                    filename = os.path.basename(file_path)
                    st.write(f"- 📁 {filename}")
                    
                    if filename.endswith('.json'):
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            st.json(data)
                        except Exception as e:
                            st.error(f"Could not read {filename}: {e}")
                    
                    elif filename.endswith('.csv'):
                        try:
                            df = pd.read_csv(file_path)
                            st.dataframe(df.head(10))
                            
                            # Simple visualization if possible
                            if 'value' in df.columns:
                                st.line_chart(df['value'])
                        except Exception as e:
                            st.error(f"Could not read {filename}: {e}")
                    
                    elif filename.endswith('.py'):
                        try:
                            with open(file_path, 'r') as f:
                                code = f.read()
                            st.code(code[:1000] + "..." if len(code) > 1000 else code, language='python')
                        except Exception as e:
                            st.error(f"Could not read {filename}: {e}")
            
            if st.button(f"❌ Close Results", key=f"close_results_{model_dir}"):
                st.session_state[f"show_results_{model_dir}"] = False
                st.rerun()

# Batch operations
st.header("🔄 Batch Operations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Convert All Notebooks"):
        for model in selected_models:
            notebook_path = os.path.join(model, "main.ipynb")
            python_path = os.path.join(model, "main_converted.py")
            
            if os.path.exists(notebook_path):
                success, result = convert_notebook_to_python(notebook_path, python_path)
                if success:
                    st.success(f"✅ {model} converted")
                else:
                    st.error(f"❌ {model} conversion failed")

with col2:
    if st.button("🚀 Run All Converted"):
        if run_mode != "Display Only":
            for model in selected_models:
                python_path = os.path.join(model, "main_converted.py")
                if os.path.exists(python_path):
                    st.session_state.execution_status[model] = "Running"
            st.rerun()
        else:
            st.info("Display Only mode - execution disabled")

with col3:
    if st.button("🔄 Reset All Status"):
        for model in available_dirs:
            if model in st.session_state.execution_status:
                del st.session_state.execution_status[model]
        st.rerun()

# Execution logs
if st.session_state.execution_logs:
    st.header("📝 Execution Logs")
    
    for model, logs in st.session_state.execution_logs.items():
        with st.expander(f"📋 {model} Logs - {logs['timestamp']}"):
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
    <p>🚀 DenseNet Notebook Runner (Simplified) | Built with Streamlit</p>
    <p>📓 Convert and execute Jupyter notebooks for hyperparameter optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
Simplified notebook runner that converts Jupyter notebooks to Python scripts for execution.

**Features:**
- Convert .ipynb to .py files
- Execute converted Python scripts
- Monitor execution status
- View results and logs

**Requirements:**
- Each model folder must contain main.ipynb
- Standard Python environment
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

st.sidebar.subheader("🔧 Process")
st.sidebar.info("""
1. **Convert**: Extract code from .ipynb
2. **Execute**: Run converted Python script
3. **Monitor**: Track execution progress
4. **Results**: View optimization outputs
""")
