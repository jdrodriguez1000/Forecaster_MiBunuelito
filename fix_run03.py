import os

path = r'c:\Users\USUARIO\Documents\Forecaster\Mi_Bunuelito\src\trainer.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix Run 03 'model_name' in append and log
    if '"model_name": f"{model_name} ({exp_name})",' in line:
        new_lines.append(line.replace('"model_name": f"{model_name} ({exp_name})",', '"model_name": f"{source_model_name} ({exp_name})",'))
    elif '"original_model": model_name,' in line and 'Run 01' not in line: # Avoid fixing Run 01 if it has similar line
         # Check if we are in Run 03 based on context (it uses base_model_key now)
         new_lines.append(line.replace('"original_model": model_name,', '"original_model": base_model_key,'))
    elif 'logger.error(f"Error in Run 03 experiment {exp_name} for {model_name}: {str(e)}")' in line:
        new_lines.append(line.replace('{model_name}', '{source_model_name}'))
    else:
        new_lines.append(line)

content = "".join(new_lines)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes for Run 03 applied.")
