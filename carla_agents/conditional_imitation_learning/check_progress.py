import os
import re

folder_path = "D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/"
files = sorted(os.listdir(folder_path))


modified_files = ([file for file in files if file.startswith('info_')])
modified_files.sort()

renamed_files_pattern = re.compile(r'(info|speedometer|left_rgb|left_semantic_segmentation|middle_rgb|middle_semantic_segmentation|right_rgb|right_semantic_segmentation)_(\d{7})_(\d{7})\\.(txt|png)')

renamed_global_steps = 0

for global_step in range(400000):  # Assuming the maximum index is 400000
    formatted_step = f"{global_step:07}"
    example_renamed_file = f"info_{formatted_step}_"
    
    if modified_files[global_step].startswith(example_renamed_file):
        renamed_global_steps += 1

total_global_steps = 400000
progress_percentage = (renamed_global_steps / total_global_steps) * 100

print(f"Total global steps: {total_global_steps}")
print(f"Renamed global steps: {renamed_global_steps}")
print(f"Progress: {progress_percentage:.2f}%")