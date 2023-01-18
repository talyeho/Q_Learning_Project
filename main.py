

import csv

import matplotlib.pyplot as plt
import numpy as np
import time
import math


# fig = plt.figure()
# ax = plt.axes(projection='3d')
height = 184
# using Anthropometric
arm_length = int(height*0.48)
# each cube is 10X10X10
# N = int(arm_length/10)              # N=8
N = 4
# for the purpose of the simulation N = 8


# # initial tables:

# rewards = for every cube in space, (we will save in the future the last 5 attempts score)
rewards = np.zeros((N*2, N*2, N))

# actions according to research book (0, 1, 2, 3, 4, 5)
actions = ['up', 'down', 'right', 'left', 'forward', 'backward']

# q_table = S*a == Q(s,a), S = # of cubes = 2N*2N*N,
q_table = np.zeros((N*2, N*2, N, len(actions)))

# feedback - x,y,z,pop,m1-5
feedback = np.zeros((N*2, N*2, N, 6))


def plotModelArg(header, table, w, h, d):
    """Use contourf to plot cube marginals"""

    x = np.indices(table.shape)[0]
    y = np.indices(table.shape)[1]
    z = np.indices(table.shape)[2]
    col = table.flatten()
    sumb = 0
    for num in col:
        sumb += num

    print(sumb)
    norm = np.linalg.norm(col)
    # col = col/norm
    # col = np.linalg.norm(col) - col
    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    plt.xlabel("common X")
    plt.ylabel("common Y")

    plt.title(header)
    p3d = ax3D.scatter3D(x, y, z, c=col, cmap="rainbow")
    clb = plt.colorbar(p3d)
    clb.ax.set_title("Balloon shows")
    plt.show()

def plotModel():
    """Use contourf to plot cube marginals"""
    data = np.zeros((N*2, N*2, N))
    for i in range(0, N*2):
        for j in range(0, N*2):
            for k in range(0, N):
                data[i, j, k] = q_table[i, j, k, np.argmax(q_table[i, j, k])]
    x = np.indices(data.shape)[0]
    y = np.indices(data.shape)[1]
    z = np.indices(data.shape)[2]
    col = data.flatten()
    norm = np.linalg.norm(col)
    # col = col/norm
    # col = np.linalg.norm(col) - col
    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    plt.xlabel("common X")
    plt.ylabel("common Y")
    plt.title("Overall max q-value for each state (cube in space)")
    p3d = ax3D.scatter3D(x, y, z, c=col, cmap="rainbow")
    plt.colorbar(p3d)
    plt.show()


def episodePlod(reward_y, episode_x):
    # plotting the points
    plt.plot(episode_x, reward_y)

    # naming the x axis
    plt.xlabel('episode')
    # naming the y axis
    plt.ylabel('reward')

    # giving a title to my graph
    plt.title('reward over episodes')

    # function to show the plot
    plt.show()

# open and make new csv file to save algorithm history
def makeCsvFile():
    header = ['episode', 'batch', 'x', 'y', 'z', 'action', 'random']
    writer=None
    try:
        writer = csv.writer(open('qllog.csv','w',encoding='UTF8', newline='\n'))
    except:
        print("Error occure while trying to open/create csv file")
    writer.writerow(header)
    return writer


# get_starting_location
#   get the location where q(s,_) is the max, which will make the agent get maximum reward
def get_starting_location():
    index = np.unravel_index(np.argmax(q_table, axis=None), q_table.shape)
    return index[0], index[1], index[2]


def get_reward(row_index, column_index, depth_index):
    movement = feedback[row_index, column_index, depth_index, 1]
    movement += feedback[row_index, column_index, depth_index, 2]
    movement += feedback[row_index, column_index, depth_index, 3]
    movement += feedback[row_index, column_index, depth_index, 4]
    movement /= 4
    # didn't pop / we should fine ture this - around 0.2-0.3?
    if feedback[row_index, column_index, depth_index, 0] == 0:
        return 0.3
    # pop
    return 1-movement



# NEED TO ADD: test if we on the edges - its a problem only in random

# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)


# get_next_action:
#   get current location and epsilon
#   the function will return the next best location to show bubble,
#   and in (1-epsilon)*100% will return random location for exploration

def get_next_action(current_row_index, current_column_index, current_depth_index, epsilon):
    # # if a randomly chosen value between 0 and 1 is less than epsilon,
    # # then choose the most promising value from the Q-table for this state.
    # if np.random.random() > epsilon:
    #     # check if work:
    #     return np.argmax(q_table[current_row_index, current_column_index, current_depth_index]), 0
    # else:  # choose a random action
    #     return np.random.randint(6), 1
    # actions according to research book (0, 1, 2, 3, 4, 5)
    # actions = ['up', 'down', 'right', 'left', 'forward', 'backward']
    #todo: dont select action that cant be made

    Action_probabilities = np.ones(6, dtype=float) * epsilon / 6
    best_action = np.argmax(q_table[current_row_index, current_column_index, current_depth_index])
    Action_probabilities[best_action] += (1.0 - epsilon)

    flag = 1
    while flag:
        action = np.random.choice(np.arange(len(Action_probabilities)), p=Action_probabilities)
        if action == 0 and current_column_index != (N*2)-1:
            flag = 0
        elif action == 1 and current_column_index != 0:
            flag = 0
        elif action == 2 and current_row_index != (N*2)-1:
            flag = 0
        elif action == 3 and current_row_index != 0:
            flag = 0
        elif action == 4 and current_depth_index != N-1:
            flag = 0
        elif action == 5 and current_depth_index != 0:
            flag = 0

    return action


# define a function that will get the next location based on the chosen action
# get_next_location:
#   get current location and the action
#   return the next location based on the action
#   If the current position is at the edges and there will be an
#   exception following the action then we will stay in the current location

def get_next_location(current_row_index, current_column_index, current_depth_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    new_depth_index = current_depth_index
    if actions[action_index] == 'up' and current_column_index < 2*N - 1:
        new_column_index += 1
    if actions[action_index] == 'down' and current_column_index > 0:
        new_column_index -= 1
    if actions[action_index] == 'right' and current_row_index < 2*N - 1:
        new_row_index += 1
    if actions[action_index] == 'left' and current_row_index > 0:
        new_row_index -= 1
    if actions[action_index] == 'forward' and current_depth_index < N - 1:
        new_depth_index += 1
    if actions[action_index] == 'backward' and current_depth_index > 0:
        new_depth_index -= 1
    return new_row_index, new_column_index, new_depth_index


# write log to csv file
def log(writer, episode, batch, row_index, column_index, depth_index, action_index, isRandom):
    str = ["" for x in range(7)]
    str[0] = episode
    str[1] = batch
    str[2] = row_index
    str[3] = column_index
    str[4] = depth_index
    str[5] = actions[action_index]
    str[6] = isRandom
    writer.writerow(str)
    # print(str, '\n')

# get_feedback:
#       upload feedback from user for each future state from csv file
#todo 7-8-9-10 is like 4-5-6
def get_feedback(episode):
    if episode >= 6:
        episode = 4
    with open('vrsim.csv', 'r') as fp:
        reader = csv.reader(fp)
        for row in list(reader)[1:]:

            if int(row[0]) == episode%10+1:
                feedback[int(row[1]), int(row[2]), int(row[3]), 0] = int(row[4])
                feedback[int(row[1]), int(row[2]), int(row[3]), 1] = float(row[5])
                feedback[int(row[1]), int(row[2]), int(row[3]), 2] = float(row[6])
                feedback[int(row[1]), int(row[2]), int(row[3]), 3] = float(row[7])
                feedback[int(row[1]), int(row[2]), int(row[3]), 4] = float(row[8])
            #   feedback[int(row[1]), int(row[2]), int(row[3]), 5] = float(row[9])


"""#### Train the AI Agent using Q-Learning"""

# define training parameters

discount_factor = 1 # discount factor for future rewards
learning_rate = 0.6
ExplorationRate = 10000
epsilon = 0.9
balloon_daily = np.zeros((N*2, N*2, N))
balloon_total = np.zeros((N*2, N*2, N))
grade = 0

def calcGrade(day, balloon_daily, epsilon, day_reward):
    # calc based on expected area here
    if day == 1 or day == 3:
        return epsilon*day_reward/60.0 + (1-epsilon)*(balloon_daily[6][5][2]+balloon_daily[6][5][3]+
                                                        balloon_daily[6][6][1]+balloon_daily[6][7][1]+
                                                        balloon_daily[7][6][1]+balloon_daily[7][7][1])/100.0
    elif day == 5:
        return epsilon*day_reward/60.0 + (1-epsilon)*(balloon_daily[6][5][2]+balloon_daily[6][5][3]+
                                                        balloon_daily[6][6][1]+balloon_daily[6][7][1]+
                                                        balloon_daily[7][6][1]+balloon_daily[7][7][1]+
                                                        balloon_daily[6][6][2] + balloon_daily[6][7][2] +
                                                        balloon_daily[7][6][2] + balloon_daily[7][7][2] +
                                                        balloon_daily[6][6][3] + balloon_daily[6][7][3] +
                                                        balloon_daily[7][6][3] + balloon_daily[7][7][3])/100.0
    else:
        return epsilon*day_reward/60.0 + (1-epsilon)*(balloon_daily[6][6][2] + balloon_daily[6][7][2] +
                                                        balloon_daily[7][6][2] + balloon_daily[7][7][2] +
                                                        balloon_daily[6][6][3] + balloon_daily[6][7][3] +
                                                        balloon_daily[7][6][3] + balloon_daily[7][7][3])/100.0

def start(writer):
    global q_table, balloon_daily, epsilon, grade

    reward_y = list(range(1, 11))
    episode_x = list(range(1, 11))
    total_reward = 0
    total_actions = 0
    last_reward = 0
    grade = 0
    # run through 10 episodes
    for episode in range(10):
        # epsilon = epsilon*(1-(episode/100))
        episode_reward = 0
        get_feedback(episode)
        # get the starting location for this episode
        row_index, column_index, depth_index = get_starting_location()
        # for each episode, we train 100 times:
        for batch in range(100):

            # # choose which action to take
            # # action_index, isRandom = get_next_action(row_index, column_index, depth_index, epsilon)
            # action_index, isRandom = get_next_action(row_index, column_index, depth_index, epsilon)
            # # save in log
            # if(writer!=None):
            #     log(writer, episode, batch, row_index, column_index, depth_index, action_index, isRandom)

            action_index = get_next_action(row_index, column_index, depth_index, epsilon)

            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index, old_depth_index = row_index, column_index, depth_index  # store the old row and column indexes
            row_index, column_index, depth_index = get_next_location(row_index, column_index, depth_index, action_index)
            total_actions += 1

            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = get_reward(row_index, column_index, depth_index)

            old_q_value = q_table[old_row_index, old_column_index, old_depth_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_table[row_index, column_index, depth_index])) - old_q_value

            # update the Q-value for the previous state and action pair
            new_q_value = (1-learning_rate)*old_q_value + (learning_rate * temporal_difference)
            q_table[old_row_index, old_column_index, old_depth_index, action_index] = new_q_value

            balloon_daily[row_index, column_index, depth_index] += 1
            balloon_total[row_index, column_index, depth_index] += 1
            episode_reward += reward
            total_reward += reward
            epsilon = max(min(ExplorationRate/(total_actions*total_reward), 0.9), 0.3)


        # epsilon = max(epsilon-0.1, 0.3)
        # for each episode, learning_rate = learning_rate - learning_rate/10 ????
        reward_y[episode] = episode_reward
        # calc grade
        if(episode+1 == 1 or episode+1 == 3 or episode+1 == 5 or episode+1 ==8):
            grade += calcGrade(episode+1, balloon_daily, epsilon, episode_reward)
        #print("reward for: " + str(episode) + "is: " + str(episode_reward))
        # plotModel()
        # q_table[5,7,2]
        plotModelArg("Day " + str(episode+1), balloon_daily, N * 2, N * 2, N)
        balloon_daily = np.zeros((N * 2, N * 2, N))
    episodePlod(reward_y, episode_x)
    plotModelArg("Total balloons show", balloon_total, N * 2, N * 2, N)




# actions according to research book (0, 1, 2, 3, 4, 5)
# actions = ['up', 'down', 'right', 'left', 'forward', 'backward']
def main():
    writer = makeCsvFile()
    grades = np.zeros((11, 11))
    for i in range(0, N*2):
        for j in range(0, N):
            q_table[0, i, j, 3] = -100
            q_table[i, 0, j, 1] = -100
            q_table[N*2-1, i, j, 2] = -100
            q_table[i, N*2-1, j, 0] = -100
    for i in range(0, N*2):
        for j in range(0, N*2):
            q_table[i, j, 0, 5] = -100
            q_table[i, j, N-1, 4] = -100
    global discount_factor, learning_rate

    # start_t = time.process_time()
    # for LR in range(1, 11):
    #     for DF in range(1, 11):
    #         learning_rate = LR/10.0
    #         discount_factor = DF/10.0
    #         for i in range(100):
    #             start(writer)
    #             grades[LR][DF] += grade
    #             print(i)
    #         grades[LR][DF] /= 100
    # print(time.process_time() - start_t)

    # maxLR=0
    # maxDF=0
    # maxGrade=0
    # for LR in range(1, 11):
    #     for DF in range(1, 11):
    #         if grades[LR][DF] > maxGrade:
    #             maxLR = LR
    #             maxDF = DF
    #             maxGrade = grades[LR][DF]

    learning_rate = 0.4
    discount_factor = 1
    start(writer)
    plotModel()



if __name__ == "__main__":
    main()
