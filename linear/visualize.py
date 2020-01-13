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


def plot_sin(epoch, loss, policy, color='#000'):
    win = 'loss'
    title = 'Loss'

    if 'loss' not in d:
        d['loss'] = {}
    if policy not in d['loss']:
        d['loss'][policy] = []

    d['loss'][policy].append((float(epoch), float(loss), color))

    # x, y = zip(*d['loss'])
    # data = [get_line(x, y, policy + ' loss', color=color, showlegend=True)]
    data = []
    for key in d['loss']:
        x, y, c = zip(*d['loss'][key])
        data.append(
            get_line(x, y, key, color=c, showlegend=True)
        )

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
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
    

