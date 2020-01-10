import numpy as np
from visdom import Visdom
import torch 
from math import isnan

viz = Visdom()

d = {}

def get_line(x, y, name, color='#000', isFilled=False, fillColor='transparent', width=2, showLegend=False): 
    if isFilled:
        fill = 'tonexty'
    else: 
        fill = 'none'

    return dict(
        x=x,
        y=y,
        mode='lines',
        tpye='custom',
        line=dict(
            color=color,
            width=width),
        fill=fill,
        fillcolor=fillColor,
        name=name,
        showlegend=showLegend
    )

def plot_loss(epoch, loss, color='#000'):
    win = 'loss'
    title = 'Loss'

    if 'loss' not in d:
        d['loss'] = []
    d['loss'].append((epoch, loss.item()))

    x,y = zip(*d['loss'])
    data = [get_line(x, y, 'loss', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'},
    )

    viz._send({'data': data, 'layout': layout, 'win': win})

def plot_reward(t, r, color='#000'):
    win = 'reward'
    title = 'Episodic Reward'

    if 'reward' not in d:
        d['reward'] = []
    d['reward'].append((t, float(r)))

    x, y = zip(*d['reward'])
    data = [get_line(x, y, 'reward', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Episodes'},
        yaxis={'title': 'Cumulative Reward'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})

def update_viz(episode, episodic_r, algo):
    global win

    if win is None: 
        win = viz.line(
            X = np.array([episode]),
            Y = np.array([episodic_r]),
            win = algo,
            opts = dict(
                title = algo,
                fillarea = False,
                xlabel = 'episode',
                ylabel = 'reward'
            )
        )
    else:
        viz.line(
            X = np.array([episode]),
            Y = np.array([episodic_r]),
            win = win,
            update = 'append'
        )
    

