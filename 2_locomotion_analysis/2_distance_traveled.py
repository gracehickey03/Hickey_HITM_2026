# Created: Jan 10, 2026
# Last modified: Feb 12, 2026
# Last git commit: Feb 12, 2026
## Updates: Cleaning data to remove the areas where p is low. 

import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import locomotion_helper_functions as lf
import cv2

## 1: Directories 
base_path = Path(__file__).parent.resolve()
dlc_dir = lf.find_parent_dir(base_path, "DLC")
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")

print(f"DLC Dir: {dlc_dir}")
print(f"Pose Estimation Dir: {pose_dir}")
print(f"Video Dir: {video_dir}")

## 2: Subjects and files
# csvs = lf.csv_list(pose_dir)

subject_list_path = results_dir.joinpath("subject_list.csv")
subjects = pd.read_csv(subject_list_path, low_memory=False)["subject"].tolist()

bodyparts = [
    "nose", "head_midpoint", "mouse_center", "tail_base", "tail3",
    "tail_end", "left_shoulder", "right_shoulder",
    "left_midside", "right_midside", "left_hip", "right_hip"
]

# subjects = subjects[0:2] # shortened list for testing 
## 3: Per-frame → per-second locomotion
for i, ind in enumerate(subjects):

    export_path = locomotion_dir / "second_movement" / f"{ind}_sec_dists.csv"
    if export_path.is_file():
        print(f"{ind}: second-by-second file already exists.")
        continue

    cup_coord_filename = (
        locomotion_dir / "roi_coordinates" / f"{ind}_cup_coords.pkl"
    )
    if not cup_coord_filename.is_file():
        print(f"{ind}: ROI not labeled. Skipping.")
        continue
    with open(cup_coord_filename, "rb") as f:
        roi_coord_data = pickle.load(f)

    # pose estimation path 
    pose_key = str(ind) + "DLC"
    matches = list(pose_dir.glob(f"*{pose_key}*.csv"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No CSV found containing '{ind}'")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple CSVs found containing '{ind}': {matches}")

    csv_path = matches[0]

    pix_per_cm_x = roi_coord_data["pix_per_cm_x"]
    pix_per_cm_y = roi_coord_data["pix_per_cm_y"]
    fps = roi_coord_data["fps"]

    video = cv2.VideoCapture(str(video_dir / f"p14_isolation_{ind}.mp4"))
    n_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    dfs_sec = []

    for part in bodyparts:
        # 1. Load coordinates and likelihood
        x, y, p = lf.import_bodypart(csv_path, part)

        # Truncate to the number of video frames (should almost always match)
        x, y, p = x[:n_video_frames], y[:n_video_frames], p[:n_video_frames]
        # csv frame count
        n_csv_frames = len(x)

        # 2. CLEANING: Mask low confidence and handle gaps
        p_threshold = 0.8
        mask = p < p_threshold
        
        x_clean = x.copy()
        y_clean = y.copy()
        x_clean[mask] = np.nan
        y_clean[mask] = np.nan

        # 3. Calculate per-frame distances (cm per frame)
        # np.diff will produce NaN if either the start or end frame is NaN
        dx = np.diff(x_clean) / pix_per_cm_x
        dy = np.diff(y_clean) / pix_per_cm_y
        frame_dists = np.sqrt(dx**2 + dy**2)
        
        # 4. BINNING: Map to real seconds
        # frame_dists is length N-1, so need to align times accordingly
        frame_times = np.arange(len(frame_dists)) / fps
        # use floor to avoid 'creating' frames 
        sec_idx = np.floor(frame_times).astype(int)

        # Replace NaNs with 0 for bincount
        # Since NaNs represent "no reliable movement," 0 is the correct sum for that frame.
        clean_frame_dists = np.nan_to_num(frame_dists, nan=0.0)
        
        sec_dists = np.bincount(sec_idx, weights=clean_frame_dists)

        n_sec = len(sec_dists)

        df_part = pd.DataFrame(
            {"distance": sec_dists},
            index=pd.MultiIndex.from_arrays(
                [
                    np.repeat(ind, n_sec),
                    np.repeat(part, n_sec),
                    np.arange(n_sec)
                ],
                names=["name", "bodypart", "second"]
            )
        )

        dfs_sec.append(df_part)

    # Save per-second distances
    dfs_sec = pd.concat(dfs_sec)
    dfs_sec.to_csv(export_path)

    print(f"{ind}: saved {n_sec/60:.2f} minutes of second-by-second movement.")
    # print(f"  len(x): {len(x)}")
    # print(f"  expected frames: {n_video_frames}")
    # print(f"  implied minutes: {len(x) / fps / 60:.2f}")

    print(f"  video frames: {n_video_frames}")
    print(f"  csv frames:   {n_csv_frames}")
    # print(f"  video minutes: {n_video_frames / fps / 60:.2f}")
    # print(f"  csv minutes:   {n_csv_frames / fps / 60:.2f}")
