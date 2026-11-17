"""
Direct notebook runner that executes all cells in FreshHarvest_AI_Project_Notebook.ipynb
without relying on ZMQ/Jupyter daemon sockets, capturing real text outputs, plots,
and tables directly into the notebook JSON.
"""

import sys
import io
import os
import json
import base64
import traceback
from pathlib import Path

# Shim for torchvision meta registrations
if 'torchvision._meta_registrations' not in sys.modules:
    import types
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(notebook_path="FreshHarvest_AI_Project_Notebook.ipynb"):
    notebook_path = Path(notebook_path)
    print(f"Executing notebook: {notebook_path.name}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    exec_env = {
        '__name__': '__main__',
        '__file__': str(notebook_path.resolve())
    }

    # Intercept display() for pandas DataFrames
    displayed_outputs = []
    def custom_display(obj):
        if hasattr(obj, 'to_html'):
            displayed_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/html": [obj.to_html() + "\n"],
                    "text/plain": [str(obj) + "\n"]
                },
                "metadata": {}
            })
        else:
            displayed_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/plain": [str(obj) + "\n"]
                },
                "metadata": {}
            })
    exec_env['display'] = custom_display

    exec_count = 1

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        raw_source = "".join(cell['source'])
        print(f"[{exec_count:02d}/{len(nb['cells'])}] Executing cell {idx}...")
        
        # Clean lines: skip shell lines starting with ! or comment them out
        cleaned_lines = []
        for line in cell['source']:
            if line.strip().startswith("!"):
                cleaned_lines.append(f"# {line}")
            else:
                cleaned_lines.append(line)
        code_to_run = "".join(cleaned_lines)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        cell_outputs = []
        captured_plots = []

        # Hook plt.show to capture image outputs
        def custom_show(*args, **kwargs):
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=130)
                img_buf.seek(0)
                b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
                captured_plots.append(b64_data)
                plt.close(fig)

        orig_show = plt.show
        plt.show = custom_show
        displayed_outputs.clear()

        try:
            exec(code_to_run, exec_env)
        except Exception as e:
            traceback.print_exc(file=stderr_buf)
            print(f"  Error in cell {idx}: {e}")
        finally:
            plt.show = orig_show
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # 1. Capture stdout
        std_out_text = stdout_buf.getvalue()
        if std_out_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + '\n' for line in std_out_text.splitlines()]
            })

        # 2. Capture stderr
        std_err_text = stderr_buf.getvalue()
        if std_err_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": [line + '\n' for line in std_err_text.splitlines()]
            })

        # 3. Add captured display outputs (DataFrames)
        cell_outputs.extend(displayed_outputs)

        # 4. Add captured plots
        for b64_img in captured_plots:
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_img,
                    "text/plain": ["<Figure size>"]
                },
                "metadata": {}
            })

        cell['execution_count'] = exec_count
        cell['outputs'] = cell_outputs
        exec_count += 1

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

    print(f"\n[DONE] Notebook execution complete. Saved outputs to {notebook_path.name}")

if __name__ == '__main__':
    run_notebook()

