import os

path = r'c:\Users\USUARIO\Documents\Forecaster\Mi_Bunuelito\src\trainer.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Run 03 'exogenous_added' -> 'exogenous_features'
    if '"exogenous_added": new_exog,' in line:
        new_lines.append(line.replace('"exogenous_added": new_exog,', '"exogenous_features": top_candidates[0]["features_used"],'))
    # Run 03 experiments setup
    elif '("Dirty (All)", list(set(base_exog + new_exog))),' in line and 'Dirty (All)' in line and 'Clean (Reduced)' not in line:
        # Check if it's the right place (Run 03/04/05 have similar setup)
        # We want to add "Stay (Previous Best)"
        new_lines.append('                ("Stay (Previous Best)", base_exog),\n')
        new_lines.append(line)
    elif 'all_results.sort(key=lambda x: x[\'mae\'])' in line:
        # We already fixed Run 01 and 02. Let's fix Run 03, 04, 05 if it says top_n or :2
        new_lines.append(line)
    else:
        new_lines.append(line)

# Let's be more precise with some global replacements
content = "".join(new_lines)

# Fix Run 03 specific:
content = content.replace(
    '            experiments = [\n                ("Dirty (All)", list(set(base_exog + new_exog))),\n',
    '            experiments = [\n                ("Stay (Previous Best)", base_exog),\n                ("Dirty (All)", list(set(base_exog + new_exog))),\n'
)

# Fix Run 04 selection (it was top_candidates = all_results[:2] or similar)
content = content.replace('top_candidates = all_results[:2]', 'top_candidates = all_results[:1]')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacement script finished.")
