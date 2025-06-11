import os
import pandas as pd

eth_base_path = "raw/"

eth_scenes = sorted([f for f in os.listdir(eth_base_path) if os.path.isdir(os.path.join(eth_base_path, f))])
eth_scene_label_map = {scene: idx for idx, scene in enumerate(eth_scenes)}

for scene in eth_scenes:
    scene_dir = os.path.join(eth_base_path, scene)
    for file_name in os.listdir(scene_dir):
        if file_name.endswith(".txt") and not file_name.startswith("crowds_zara03") and not file_name.endswith("obsmat.txt"):
            file_path_txt = os.path.join(scene_dir, file_name)
            try:
                df = pd.read_csv(file_path_txt, delim_whitespace=True, header=None)
                df.columns = ["frame", "track_id", "x", "y"]

                df_clean = df[["frame", "track_id", "x", "y"]].copy()
                df_clean["label_num"] = -1  #No labels in eth
                df_clean["scene_id"] = eth_scene_label_map[scene]

                output_file = os.path.splitext(file_name)[0] + "_clean.txt"
                output_path = os.path.join(scene_dir, output_file)
                df_clean.to_csv(output_path, sep=' ', index=False, header=False)
            except Exception as e:
                print(f"Failed to process {file_path_txt}: {e}")
        elif file_name.endswith("obsmat.txt"):
            file_path_txt = os.path.join(scene_dir, file_name)
            try:
                df = pd.read_csv(file_path_txt, delim_whitespace=True, header=None)
                df.columns = ["frame", "track_id", "x","z", "y", "vx", "vy", "vz"]

                df_clean = df[["frame", "track_id", "x", "y"]].copy()
                df_clean["label_num"] = -1  # No labels in eth
                df_clean["scene_id"] = eth_scene_label_map[scene]

                output_file = os.path.splitext(file_name)[0] + "_clean.txt"
                output_path = os.path.join(scene_dir, output_file)
                df_clean.to_csv(output_path, sep=' ', index=False, header=False)
            except Exception as e:
                print(f"Failed to process {file_path_txt}: {e}")


