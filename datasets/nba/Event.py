from Constant import Constant
from Moment import Moment
from Team import Team
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
import numpy as np

class Event:
    """A class for handling and showing events"""

    def __init__(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values))

    def get_continues_traj(self):
        moment_length = len(self.moments)
        unix_timestemps =[]
        all_player_locations = []
        for i in range(moment_length):
            player_locations = []
            cur_moment = self.moments[i]
            if len(cur_moment.players) < 10:
                continue
            for k in range(10):
                player_locations.append([cur_moment.players[k].x, cur_moment.players[k].y])
            player_locations.append([cur_moment.ball.x, cur_moment.ball.y])
            unix_timestemps.append(cur_moment.unix)
            all_player_locations.append(player_locations)

        # print(len(unix_timestemps) , " unix")
        # print("players ", np.array(all_player_locations).shape)
        return unix_timestemps, all_player_locations

    def get_longest_traj(self):
        moment_length = len(self.moments)
        print(moment_length)


    def get_traj(self):
        moment_length = len(self.moments)
        traj_num = moment_length // 150 # 50 + 100, past 50(0,04s, 2s), future 100 (0.04s, 4s)
        all_all_player_locations = [] #(N,15,11,2)
        for i in range(traj_num):
            all_player_locations = [] # (15,11,2)
            # check if is 10 people
            flag = True
            for j in range(15):
                time_stamp = 150 * i + 10 * j
                cur_moment = self.moments[time_stamp]
                if len(cur_moment.players) < 10:
                    flag = False
            if not flag:
                continue
            # check if the same people 
            flag = True
            time_stamp1 = 150 * i
            time_stamp2 = 150 * i + 140
            cur_moment1 = self.moments[time_stamp1]
            cur_moment2 = self.moments[time_stamp2]
            for j in range(10):
                if cur_moment1.players[j].id != cur_moment2.players[j].id:
                    flag = False
            if not flag:
                continue

            time_stamp1 = 150 * i
            time_stamp2 = 150 * i + 140
            if self.moments[time_stamp2].game_clock - self.moments[time_stamp1].game_clock < -5.7 or self.moments[time_stamp2].game_clock - self.moments[time_stamp1].game_clock> -5.5:
                continue

            for j in range(15):
                time_stamp = 150 * i + 10 * j
                cur_moment = self.moments[time_stamp]
                player_locations = []  #(11,2)
                for k in range(10):
                    player_locations.append([cur_moment.players[k].x,cur_moment.players[k].y])
                player_locations.append([cur_moment.ball.x,cur_moment.ball.y])
                all_player_locations.append(player_locations)
            all_all_player_locations.append(all_player_locations)
        all_all_player_locations = np.array(all_all_player_locations,dtype=np.float32)
        del_list = []
        # check if the traj contiguous
        for i in range(len(all_all_player_locations)):
            seq_data = all_all_player_locations[i] #(15,11,2)
            diff_v = seq_data[1:,:-1,:] - seq_data[:-1,:-1,:]
            diff_a = diff_v[1:,:,:] - diff_v[:-1,:,:]
            diff_v = np.linalg.norm(diff_v,ord=2,axis=2)
            diff_a = np.linalg.norm(diff_a,ord=2,axis=2)
            if np.max(diff_v) >= 9 or np.max(diff_a) >= 5: # people cannot move that fast
                del_list.append(i)
        all_all_player_locations = np.delete(all_all_player_locations,del_list,axis=0)

        # check if the ball out of court
        del_list = []
        for i in range(len(all_all_player_locations)):
            seq_data = all_all_player_locations[i] #(15,11,2)
            ball_x = seq_data[:,-1,0]
            ball_y = seq_data[:,-1,1]
            if np.max(ball_x) > Constant.X_MAX - Constant.DIFF or np.min(ball_x) < 0 or np.max(ball_y) > Constant.Y_MAX or np.min(ball_y) < 0:
                del_list.append(i)
        all_all_player_locations = np.delete(all_all_player_locations,del_list,axis=0)
        return all_all_player_locations

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def show(self):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        table.scale(1, Constant.SCALE)
        # table_cells = table.properties()['child_artists']
        # for cell in table_cells:
        #     cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info),
                         frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        anim.save('example.gif',writer='imagemagick')
        # plt.show()
