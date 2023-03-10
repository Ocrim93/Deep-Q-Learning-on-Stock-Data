from plotly.graph_objs import *
from plotly.offline import  iplot
import datetime
import pandas_datareader as web
from plotly import tools

# train_test_array  = [train_array,test_array] 
# date_split = the date that separates train from  test
def plot_train_test(train_test_array,date_split):
	data = []
	for frame in train_test_array:
		data.append(Candlestick(x= frame.index, open = frame['Open'],high = frame['High'],low = frame['Low'],close = frame['Close'], name = frame.name)) 
	layout = {
         'shapes': [
             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
         ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
        ]
    }
	figure = Figure(data=data, layout=layout)
	iplot(figure)

def plot_loss_reward(total_losses,total_rewards):

    figure = tools.make_subplots(rows=1,cols=2,subplot_titles=('loss','reward'),print_grid=False)
    figure.append_trace(Scatter(y=total_losses,mode='lines',line=dict(color='skyblue')),1,1)
    figure.append_trace(Scatter(y=total_losses,mode='lines',line=dict(color='orange')),1,2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400,width = 900,showlegend =True)
    iplot(figure)

def retrieveData(stock_name, years):
    end = datetime.datetime.now()
    start = datetime.datetime(end.year - years,end.month,end.day)
    df = web.DataReader(stock_name,data_source="yahoo",start=start, end=end)

    #save the dataframe
    df.to_csv('Stocks/'+stock_name+'.csv')

  
