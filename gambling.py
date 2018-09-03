import numpy as np
WINNING_PROB = 0.4
WIN = 0
LOSS = 1
WIN_REWARD =  1
LOSS_REWARD = -1

def randomBet(current):
	max_bet = min(current, 100 - current)
	return np.random.choice(range(1, max_bet + 1))

def toss():
	if np.random.random() <= WINNING_PROB:
		return WIN
	else:
		return LOSS

def play(initialState):
	current = initialState
	trajectory = []
	while current != 0 and current != 100:
		bet = randomBet(current)
		trajectory.append((current, bet))
		if toss() == WIN:
			current += bet
		else:
			current -= bet
	if current == 0:
		return LOSS_REWARD, trajectory
	else:
		return WIN_REWARD, trajectory

def monteCarlo(nEpisodes):
	stateActionValues = np.zeros((101, 51))
	stateActionCount = np.ones((101, 51))

	for episode in range(nEpisodes):
		if episode % 1000 == 0:
			print(episode)

		initialState = np.random.choice(range(1, 100))
		reward, trajectory = play(initialState)
		for state, action in trajectory:
			stateActionValues[state, action] += reward
			stateActionCount[state, action] += 1
	return stateActionValues/stateActionCount

def run():
	stateActionValues = monteCarlo(5000000)
	actions = np.zeros(100)
	values = np.zeros(100)
	for i in range(100):
		actions[i] = np.argmax(stateActionValues[i, :])
		values[i] = np.max(stateActionValues[i, :])
	print(values)
	print(actions)

run()






























