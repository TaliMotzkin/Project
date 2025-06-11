import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant
import numpy as np


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        all_trajs = []
        for i in range(last_default_index):
            event = data_frame['events'][i]
            self.event = Event(event)
            trajs = self.event.get_traj()  # (N,15,11,2)
            if len(trajs) > 0:
                all_trajs.append(trajs)
                # print(i,len(trajs))
        if len(all_trajs) > 0:
            all_trajs = np.concatenate(all_trajs,axis=0)
        return all_trajs


    def printing_legnths(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        for i in range(last_default_index):
            event = data_frame['events'][i]
            self.event = Event(event)
            self.event.get_longest_traj()  # (N,15,11,2)


    def read_json_continues(self):
        data_frame = pd.read_json(self.path_to_json)
        all_trajs = []
        all_unix = []
        for i in range(2, 150):
            # print("iteration {}".format(i))
            event = data_frame['events'][i]
            self.event = Event(event)
            unix, players = self.event.get_continues_traj()
            if len(unix) == 0:
                continue
            all_trajs.extend(players)
            all_unix.extend(unix)

        print("all_trajs", np.array(all_trajs).shape)
        print("all_unix", np.array(all_unix).shape)
        # Step 1: Pair and sort by Unix timestamp
        paired = list(zip(all_unix, all_trajs))
        paired.sort(key=lambda x: x[0])  # sort by unix timestamp

        # Step 2: Separate sorted lists
        sorted_unix = [p[0] for p in paired]
        sorted_trajs = [p[1] for p in paired]

        # Step 3: Remove duplicate timestamps
        all_unix_np = np.array(sorted_unix)
        unique_unix, counts = np.unique(all_unix_np, return_counts=True)
        duplicate_unix = set(unique_unix[counts > 1])

        cleaned_trajs = [
            traj for traj, t in zip(sorted_trajs, sorted_unix) if t not in duplicate_unix
        ]

        cleaned_trajs = np.array(cleaned_trajs)[::10]  # shape: (num_traj, 11, 2)
        print("Cleaned traj shape:", cleaned_trajs.shape)

        # Step 4: Chunk into (N, 15, 11, 2)
        total_frames = cleaned_trajs.shape[0]
        N = total_frames // 15
        chunked = cleaned_trajs[:N * 15].reshape(N, 15, 11, 2)
        print("Final chunked shape:", chunked.shape)

        return chunked



