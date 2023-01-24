class Environment:

	def __init__(self,data,history_t=90):
		self.data = data
		self.history_t = history_t
		self.reset()

	def reset(self):
		self.t = 0
		self.done = False
		self.profits = 0
		self.positions = []
		self.position_value =0
		self.history = [0 for _ in range(self.history_t)]
		return [self.position_value] + self.history


	def step(self,act):
		reward = 0 

		''' if act = 
			0 :stay,
			1 : buy,
			2 : sell
		'''
		if act == 1: # buy / append the closing price at index (time) self.t
			self.positions.append(self.data.iloc[self.t,:]['Close'])
		elif act == 2: 	# sell	
			if len(self.positions) == 0:
				reward = -1
			else:
				profits = 0
				for p in self.positions:
					profits += (self.data.iloc[self.t,:]['Close'] - p ) # difference of the selling and buying closing price
				reward += profits		
				self.profits += profits
				self.positions = []
		#set next time 
		self.t += 1
		self.position_value = 0		
		for p in self.positions:
			self.position_value += (self.data.iloc[self.t,:]['Close'] - p ) # self.positions is only non-empty when you did not sell yet 
		self.history.pop(0)
		self.history.append(self.data.iloc[self.t,:]['Close'] - self.data.iloc[(self.t-1),:]['Close'] ) #difference between the buying close price and the next time close price

		# clipping reward
		if reward > 0: 
			reward = 1
		elif reward < 0 :
			reward = -1 


		return [self.position_value] + self.history, reward,self.done   # obs, reward, done		







