from pathlib import Path
import os

import torch
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print(f"Current CUDA device ID: {device_id}")
else:
    raise Exception("GPU not found!")

try:
    import deeplabcut
except Exception as e:
    print("Error importing deeplabcut!")
    print(e)

    
print("Dependencies successfully imported!\n")


working_dir = Path("") ## path to parent DLC directory 

video_folder = working_dir.joinpath("") ## path to video directory 

video_list = []
for filename in os.listdir(video_folder):
    if filename.endswith((".mp4", ".asf", ".mov")): 
        video_list.append(os.path.join(video_folder, filename))

for i, vid in enumerate(video_list): 
    print(f"{i}: {vid}")



project_dir = working_dir.joinpath("Projects")
print(f"Project directory: {project_dir}\n")

project_name = "" ## project folder name 

project_path = project_dir.joinpath(project_name)
print(f"Project path: {project_path}\n")

# ensure that project path starts with /blue/adamdewan, NOT /blue_adamdewan
config_path = project_path.joinpath("config.yaml")
print(f"Project config: {config_path}\n")

print("Now analyzing videos:\n")
deeplabcut.analyze_videos(
    config_path,
    videos=video_list, 
    shuffle=1,
    save_as_csv=True,
    batch_size=8
)