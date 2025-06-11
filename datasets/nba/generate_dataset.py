
from Game import Game
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')

args = parser.parse_args()

data_root = '../nba/source/nba_data_extracted'
data_target = '../nba/target'
if not os.path.exists(data_target):
    os.mkdir(data_target)
json_list = os.listdir(data_root)
print(json_list)
all_trajs = []
iteration = 0
for file_name in json_list:
	if '.json' not in file_name:
		continue
	print("iter" , iteration)
	json_path = data_root + '/' + file_name
	game = Game(path_to_json=json_path)
	trajs = game.read_json()
	if len(trajs) == 0:
		continue
	# all_trajs = game.read_json_continues()
	trajs = np.unique(trajs,axis=0)
	print(trajs.shape)
	all_trajs.append(trajs)
	iteration += 1

all_trajs = np.concatenate(all_trajs,axis=0)
all_trajs = np.unique(all_trajs,axis=0)
print(len(all_trajs))

index = list(range(len(all_trajs)))
from random import shuffle
shuffle(index)
train_set_1 = all_trajs[index[:35000]]#37500
train_set_2 = all_trajs[index[35000:70000]]
train_set_3 = all_trajs[index[70000:105000]]
test_set = all_trajs[index[105000:]]
# print('train num:',train_set.shape[0])
# print('test num:',test_set.shape[0])
#
np.save(data_target+'/train_trial_1.npy',train_set_1)
np.save(data_target+'/train_trial_2.npy',train_set_2)
np.save(data_target+'/train_trial_3.npy',train_set_3)
np.save(data_target+'/test_trial.npy',test_set)

