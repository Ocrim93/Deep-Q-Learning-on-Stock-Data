''' 
	DQN - Deep Q Learning --> 
	Deep Learning to Reinforcement Learning 
	Q-Learning a memory table Q(S,A) where S is the state and A is the action will make or made by the agent

	Chain enables us to write a neural net based on composition, 
	without bothering about routine works like collecting parameters, 
	serialization, copying the structure with parameters shared, etc.

'''
import chainer
import chainer.links as L 
import chainer.functions as F
import time
import numpy as np
import copy

def train_DQN(env):

	class Q_network(chainer.Chain):

		def __init__(self,input_size,hidden_size,output_size):
			# constructor inheritance from the base class (chainer.Chain)  
			super(Q_network,self).__init__(
				fc1 = L.Linear(input_size,hidden_size),
				fc2 = L.Linear(hidden_size,hidden_size),
				fc3 = L.Linear(hidden_size,output_size)
				)

		def __call__(self,x):
			h= F.relu(self.fc1(x))
			h= F.relu(self.fc2(h))
			y = self.fc3(h)
			return y

		def reset(self):
			self.zerograds()	

	Q = Q_network(input_size = env.history_t +1,hidden_size = 100,output_size =3)
	Q_ast = copy.deepcopy(Q)

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(Q)

	epoch_num = 50
	step_max = len(env.data) -1 
	memory_size = 200
	batch_size = 20
	epsilon = 1.0
	epsilon_decrease = 1e-3
	epsilon_min = 0.1
	start_reduce_epsilon = 200
	train_freq = 10
	update_q_freq = 20
	gamma = 0.97
	show_log_freq = 5

	memory = []
	total_step = 0 
	total_rewards = []
	total_losses = []

	start = time.time()
	for epoch in range(epoch_num):
		
		# Resetting the parameter for each epoch
		
		pobs = env.reset()
		step = 0 
		done = False
		total_reward = 0
		total_loss = 0

		while not done and step < step_max:

			#select act
			pact = np.random.randint(3)
			if np.random.rand() > epsilon:
				pact = Q(np.array(pobs,dtype=np.float32).reshape(1,-1)) # if you do not know the dimension for the reshaping put -1
																		# shape is view not a copy, try.base Base object if memory is from some other object. otherwise None is returned 
				pact = np.argmax(pact.data)  # Returns the indices of the maximum values along an axis.

			#act 	
			obs,reward, done = env.step(pact)

			#add 
			memory.append((pobs,pact,reward,obs,done))
			if len(memory) > memory_size:
				memory.pop(0)

			# train or update q 

			if len(memory) == memory_size:
				if total_step % train_freq == 0:
					shuffled_memory= np.random.permutation(memory) #Randomly permute a sequence, or return a permuted range
					memory_idx = range(len(shuffled_memory))
					for i in memory_idx[::batch_size]:
						batch = np.array(shuffled_memory[i:i+batch_size])
						b_pobs = np.array(batch[:,0].tolist(),dtype=np.float32).reshape(batch_size,-1)
						b_pact = np.array(batch[:,1].tolist(),dtype=np.int32)
						b_reward = np.array(batch[:,2].tolist(),dtype=np.int32)
						b_obs = np.array(batch[:,3].tolist(),dtype=np.float32).reshape(batch_size,-1)
						b_done = np.array(batch[:,4].tolist(),dtype=np.bool)

						q = Q(b_pobs)
						maxq = np.max(Q_ast(b_obs).data,axis=1)
						target = copy.deepcopy(q.data)
						for j in range(batch_size):
							target[j,b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
						Q.reset()
						
						loss= F.mean_squared_error(q,target)
						total_loss += loss.data
						loss.backward()
						optimizer.update()
				if total_step % update_q_freq ==0 :
					Q_ast = copy.deepcopy(Q)

			#epsilon
				
			if epsilon > epsilon_min and total_step > start_reduce_epsilon:
				epsilon -= epsilon_decrease 

			#next step
			total_reward += reward 
			pobs = obs
			step += 1
			total_step += 1


		total_rewards.append(total_reward)
		total_losses.append(total_loss)


		if (epoch+1) % show_log_freq == 0 :
			log_reward = sum(total_rewards[((epoch+1)- show_log_freq ):])/show_log_freq	
			log_loss = sum(total_losses[((epoch+1)- show_log_freq ):])/show_log_freq		
			elapsed_time = time.time()-start
			print('\t'.join(map(str,[epoch+1,epsilon,total_step,log_reward,log_loss,elapsed_time])))
			start= time.time()

	return Q,total_losses,total_rewards






















