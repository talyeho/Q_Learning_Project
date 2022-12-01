

import numpy as np
import csv

height = 184
# using Anthropometric
arm_length = int(height*0.48)
# each cube is 10X10X10
N = int(arm_length/10)              # N=8

# for the purpose of the simulation N = 8
N = 8

# # initial tables:

# rewards = for every cube in space, (we will save in the future the last 5 attempts score)
rewards = np.zeros((N*2, N*2, N))
success = np.full((N*2, N*2, N), 1)
max_reward = 1

for x in range(N, 2*N):
    for y in range(N, 2*N):
        success[x, y, 0] = 0

# actions according to research book (0, 1, 2, 3, 4, 5)
actions = ['up', 'down', 'right', 'left', 'forward', 'backward']

# q_table = S*a == Q(s,a), S = # of cubes = 2N*2N*N,
q_table = np.zeros((N*2, N*2, N, len(actions)))

# feedback - x,y,z,pop,m1-5
feedback = np.zeros((N*2, N*2, N, 6))


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


# get_reward:
#   receive the location of the bubble, then extract: if the bubble pop, and the movement
#   the reward will calculate as the follow:
#   if pop = 0
#       reward = 0.2
#   else
#       if movement > 0.9
#           reward = 0.2
#       else
#           maximum reward
# we like to make the algorithm to show balloons where the user can pop it but with poor movement
# todo
def get_reward(row_index, column_index, depth_index):
    movement = feedback[row_index,column_index,depth_index, 1]
    movement += feedback[row_index, column_index, depth_index, 2]
    movement += feedback[row_index, column_index, depth_index, 3]
    movement += feedback[row_index, column_index, depth_index, 4]
    movement += feedback[row_index, column_index, depth_index, 5]
    movement /= 5
    if feedback[row_index,column_index,depth_index, 0] == 0:
        return 0.2
    if movement>0.9:
        return 0.2
    return 1



# NEED TO ADD: test if we on the edges - its a problem only in random

# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)


# get_next_action:
#   get current location and epsilon
#   the function will return the next best location to show bubble,
#   and in (1-epsilon)*100% will return random location for exploration

def get_next_action(current_row_index, current_column_index, current_depth_index, epsilon):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        # check if work:
        return np.argmax(q_table[current_row_index, current_column_index, current_depth_index]), 0
    else:  # choose a random action
        return np.random.randint(6), 1


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
    if actions[action_index] == 'up' and current_row_index < 2*N - 1:
        new_row_index += 1
    elif actions[action_index] == 'down' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < 2*N - 1:
        new_column_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    elif actions[action_index] == 'forward' and current_depth_index < N - 1:
        new_depth_index += 1
    elif actions[action_index] == 'backward' and current_depth_index > 0:
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
    print(str, '\n')

# get_feedback:
#       upload feedback from user for each future state from csv file
def get_feedback(episode):
    with open('feedback.csv', 'r') as fp:
        reader = csv.reader(fp)
        for row in list(reader)[1:]:
            if int(row[0]) == episode:
                feedback[int(row[1]), int(row[2]), int(row[3]), 0] = int(row[4])
                feedback[int(row[1]), int(row[2]), int(row[3]), 1] = float(row[5])
                feedback[int(row[1]), int(row[2]), int(row[3]), 2] = float(row[6])
                feedback[int(row[1]), int(row[2]), int(row[3]), 3] = float(row[7])
                feedback[int(row[1]), int(row[2]), int(row[3]), 4] = float(row[8])
                feedback[int(row[1]), int(row[2]), int(row[3]), 5] = float(row[9])


"""#### Train the AI Agent using Q-Learning"""

# define training parameters

discount_factor = 0.9  # discount factor for future rewards
learning_rate = 1
epsilon = 0.9

def start(writer):
    # run through 1000 episodes
    for episode in range(1000):
        get_feedback(episode)
        # get the starting location for this episode
        row_index, column_index, depth_index = get_starting_location()

        # for each episode, we train 100 times:
        for batch in range(1000):

            # choose which action to take
            action_index, isRandom = get_next_action(row_index, column_index, depth_index, epsilon)

            # save in log
            if(writer!=None):
                log(writer, episode, batch, row_index, column_index, depth_index, action_index, isRandom)

            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index, old_depth_index = row_index, column_index, depth_index  # store the old row and column indexes
            row_index, column_index, depth_index = get_next_location(row_index, column_index, depth_index, action_index)

            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = get_reward(row_index, column_index, depth_index)
            old_q_value = q_table[old_row_index, old_column_index, old_depth_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_table[row_index, column_index, depth_index])) - old_q_value

            # update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_table[old_row_index, old_column_index, old_depth_index, action_index] = new_q_value


        # for each episode, learning_rate = learning_rate - learning_rate/10 ????
        # learning_rate = learning_rate - learning_rate/10


def main():
    writer = makeCsvFile()
    start(writer)

if __name__ == "__main__":
    main()
