import json

ATTS_TO_KEEP = ['boundary_node', 'VTD', 'TOPPOP', 'PRES16D', 'PRES16R', 'pos', 'id']

data = {}
with open('TX_vtds.json', 'r') as f1:
    data = json.load(f1)

for i, n in data['nodes']:
    data['nodes'][i] = {k: n[k] for k in ATTS_TO_KEEP}

with open('TX_vtds_clean.json', 'w') as f2:
    json.dump(data, f2, indent=2)
