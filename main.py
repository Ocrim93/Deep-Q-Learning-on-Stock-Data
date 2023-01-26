import numpy as np
import pandas as pd
import subprocess     # Subprocess is a python module that is used to run new codes by creating new processes.
					  # It allows the user to create a new application within the currently executing python program.	



from plotly.graph_objs import *
from plotly.offline import init_notebook_mode,iplot_mpl
import utilities
import EnvironmentClass
import DQN
#init_notebook_mode()

calling_output = subprocess.check_output(['ls','-la'])
print(calling_output)

years = 2 
stock_name = 'AAPL'
# Get the stock quote

#utilities.retrieveData(stock_name,years)


data = pd.read_csv('Stocks/'+stock_name+'.csv')
data['Date'] = pd.to_datetime(data['Date'])  # when a csv file is imported and a Data Frame is made, the Data time objects in the file are read as a string 
											# and not a Datetime object

data = data.set_index('Date')

train ,test = np.array_split(data,2)
train.name = 'train'
test.name = 'test'

#utilities.plot_train_test([train,test],train.index[-1])
print(np.random.rand())

env = EnvironmentClass.Environment(train)

Q,total_losses,total_rewards = DQN.train_DQN(env)
utilities.plot_loss_reward(total_losses,total_rewards)

'''
for _ in range(3):
	pact = np.random.randint(3) # integer [0,1,2]
	print(pact)
	print(env.step(pact))




'''


