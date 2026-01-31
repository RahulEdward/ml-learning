import json
from pathlib import Path

# Path to the notebook
nb_path = Path(r'd:/machine-learning-for-trading-main/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/01_parse_itch_order_flow_messages.ipynb')
output_path = Path(r'd:/machine-learning-for-trading-main/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/markdown_content.json')

# Read the notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract markdown cells
markdown_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        # Store index to put it back later
        markdown_cells.append({
            'index': i,
            'source': cell['source']
        })

# Write to JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(markdown_cells, f, indent=2)

print(f"Extracted {len(markdown_cells)} markdown cells to {output_path}")
