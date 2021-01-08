import numpy as np
from math import isnan
from visdom import Visdom
import torch 

viz = Visdom()

d = {}
win = None

def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
    if isFilled:
        fill = 'tonexty'
    else:
        fill = 'none'

    return dict(
        x=x,
        y=y,
        mode='lines',
        type='custom',
        line=dict(
            color=color,
            width=width),
        fill=fill,
        fillcolor=fillcolor,
        name=name,
        showlegend=showlegend
    )


def plot_loss(epoch, loss, policy, algo, color='#000'):
    win = algo + 'loss'
    title = algo + ' Loss'

    if algo not in d:
        d[algo] = {}
    if policy not in d[algo]:
        d[algo][policy] = []

    d[algo][policy].append((epoch, loss.item(), color))

    # x, y = zip(*d['loss'])
    # data = [get_line(x, y, policy + ' loss', color=color, showlegend=True)]
    data = []
    for key in d[algo]:
        x, y, c = zip(*d[algo][key])
        
        data.append(
            get_line(x, y, key, color=c[0], showlegend=True)
        )

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot_reward(t, r, algo, color='#000'):
    win = algo + 'reward'
    title = algo + ' Episodic Reward'

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
    

