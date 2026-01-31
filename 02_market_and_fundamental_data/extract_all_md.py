import json
from pathlib import Path
import os

# Root directory
root_dir = Path(r'd:/machine-learning-for-trading-main/02_market_and_fundamental_data')
output_path = root_dir / 'all_notebook_markdown.json'

all_data = {}

# Walk through all directories
for path in root_dir.rglob('*.ipynb'):
    # Skip already translated files or checkpoints
    if '_HINDI' in path.name or '.ipynb_checkpoints' in str(path):
        continue
    
    # Read the notebook
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue

    # Extract markdown cells
    markdown_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            markdown_cells.append({
                'index': i,
                'source': cell['source']
            })
    
    # Store with relative path as key
    rel_path = str(path.relative_to(root_dir))
    all_data[rel_path] = markdown_cells
    print(f"Processed {rel_path}: {len(markdown_cells)} markdown cells")

# Write to JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2)

print(f"Saved extracted markdown from {len(all_data)} notebooks to {output_path}")
