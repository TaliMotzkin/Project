import os
import pandas as pd

file_path = "raw/"

label_map = {
    "Biker": 0,
    "Pedestrian": 1,
    "Skater": 2,
    "Cart": 3,
    "Car": 4,
    "Bus": 5,
}

scene_folders = sorted([f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))])
scene_label_map = {scene: idx for idx, scene in enumerate(scene_folders)}

all_data = []

for scene in scene_folders:
    scene_dir = os.path.join(file_path, scene)
    for file_name in os.listdir(scene_dir):
        if file_name.endswith(".txt") and not file_name.endswith("clean.txt"):
            file_path_txt = os.path.join(scene_dir, file_name)
            try:
                df = pd.read_csv(file_path_txt, sep=' ', header=None)
                df.columns = ["id",  "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded",
                              "generated", "label"]
                # remove lost objects
                df = df[df["lost"] == 0]

                #center of bounding box
                df["x"] = (df["xmin"] + df["xmax"]) / 2
                df["y"] = (df["ymin"] + df["ymax"]) / 2

                # label and scene
                df["label_num"] = df["label"].str.replace('"', '').map(label_map)
                df["scene_id"] = scene_label_map[scene]

                df_clean = df[["frame", "id", "x", "y"]]
                # df_clean = df[["frame", "id", "x", "y", "label_num", "scene_id"]]

                output_file = os.path.splitext(file_name)[0] +scene+ "V2.txt"
                output_path = os.path.join(scene_dir, output_file)
                df_clean.to_csv(output_path, sep=' ', index=False, header=False)
            except Exception as e:
                print(f"Failed to process {file_path_txt}: {e}")

