#!/usr/local/bin/env python
from __future__ import division
import sys
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, messages, analyze
from radd import theta
from IPython.display import HTML, Javascript, display

class PBinJ(object):
    """ animated ProgressBar (PB) to be used (inJ)upyter notebooks
    (set bartype to 'uglybar' if running from terminal)
    """
    def __init__(self, bartype='colorbar', n=None, color='blue', title='Progress'):
        colors = {'green': '#16a085', 'blue': '#3A539B', 'red': "#e74c3c"}
        self.color = colors[color]
        self.n=n
        self.displayed=False
        self.bartype=bartype
        self.title=title
        self.init_bars()

    def init_bars(self):
        if self.bartype=='uglybar':
            import progressbar
            if self.n is not None:
                self.bar = progressbar.ProgressBar(0, self.n)
            self.new_prog_string = ''.join([self.title, ': {0}'])
            self.update = self.update_uglybar
        else:
            import uuid
            self.barid=str(uuid.uuid4())
            if self.bartype=='colorbar':
                args = [self.title, self.color, "500", self.barid, '0%', 'left']
                self.update = self.update_colorbar
            else:
                args = [self.title, self.color, "105", self.barid, '100%', 'center']
                self.update = self.update_progress
            self.bar="""<div<p>{0}</p> <div style="border: 1px solid {1}; width:{2}px">
            <div id="{3}" style="background-color:{1}; width:{4}; text:''; color:#fff; text-align:{5};">
            &nbsp;</div> </div> </div>""".format(*args)

    def display_bars(self):
        if self.bartype=='uglybar':
            self.bar.start()
        else:
            display(HTML(self.bar))
        self.displayed=True

    def update_progress(self, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        display(Javascript("$('div#{}').text({:.5f})".format(self.barid, new_progress)))

    def update_colorbar(self, i=None, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        if i is not None:
            display(Javascript("$('div#{}').width('{:.2f}%')".format(self.barid, ((i+1)*1./(self.n+1))*100)))
        if new_progress is not None:
            self.update_progress(new_progress)

    def update_uglybar(self, i=None, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        if new_progress is not None:
            sys.stdout.write('\r'+self.new_prog_string.format(str(new_progress)))
            sys.stdout.flush()
        if i is not None:
            self.bar.update(i)

    def clear(self):
        if self.bartype=='uglybar':
            sys.stdout.flush
        else:
            from IPython.display import clear_output
            clear_output()


class NestedProgress(object):
    """ initialize multiple progress bars for tracking nested stages of fitting routine
    """
    def __init__(self, name='inits_bar', bartype='colorbar', n=None, init_state=None, color='blue', title='global fmin'):
        self.bars = {}
        self.history = []
        self.add_bar(name=name, bartype=bartype, n=n, init_state=init_state, color=color, title=title)

    def add_bar(self, name='inits_bar', bartype='colorbar', n=None, init_state=None, color='blue', title='global fmin'):
        bar = PBinJ(bartype=bartype, n=n, color=color, title=title)
        self.bars[name] = bar
        if init_state is not None:
            self.reset_bar(name, init_state)

    def reset_bar(self, name, init_state=None):
        self.history = [init_state]
        self.bars[name].update(new_progress=init_state)

    def update(self, name='all', i=None, new_progress=None):
        if name=='all':
            update_list = listvalues(self.bars)
        else:
            update_list = [self.bars[name]]
        for bar in update_list:
            if bar.bartype=='infobar' and new_progress:
                bar.update(new_progress)
            elif bar.bartype=='colorbar':
                bar.update(i=i, new_progress=new_progress)
            else:
                continue

    def callback(self, x, fmin, accept):
        """ A callback function for reporting basinhopping status
        Arguments:
            x (array):
                parameter values
            fmin (float):
                function value of the trial minimum, and
            accept (bool):
                whether or not that minimum was accepted
        """
        if fmin <= np.min(self.history):
            self.bars['glb_basin'].update(new_progress=fmin)
        if accept:
            self.history.append(fmin)
            self.bars['lcl_basin'].update(new_progress=fmin)

    def clear(self):
        for bar in listvalues(self.bars):
            bar.clear()

def render_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def animate_dpm(model):
    """ to render animation within a notebook :
        vis.render_animation(vis.animated_dpm_example(MODEL))
    """
    from matplotlib import animation
    params = deepcopy(model.inits)
    bound=theta.scalarize_params(params)['a']
    x, gtraces, straces, xi, yi, nframes = gen_re_traces(model, params)
    f, axes = build_decision_axis(onset=x[0][0], bound=bound)
    glines = [axes[i].plot([], [], linewidth=1.5)[0] for i, n in enumerate(gtraces)]
    slines = [axes[i].plot([xi[i]], [yi[i]], linewidth=1.5)[0] for i, n in enumerate(straces)]
    f_args = (x, gtraces, glines, straces, slines, params, xi, yi)
    anim=animation.FuncAnimation(f, re_animate_multiax, fargs=f_args, frames=nframes, interval=1, blit=True)
    return anim

def gen_re_traces(model, params):
    sim = deepcopy(model.opt.simulator)
    sim.__update_steps__(dt=.001)
    ssd, nssd, nss, nss_per, ssd_ix = sim.ssd_info
    nsstot = nss * nssd
    params = sim.vectorize_params(params)
    bound = params['a'][0]
    Pg, Tg, xtb = sim.__update_go_process__(params)
    Ps, Ts = sim.__update_stop_process__(params)
    Ts=[Ts[0, i] for i in [2, -1]]
    dvg, dvs = sim.sim_fx(params, analyze=False)
    dvg = dvg[0, :nss, :].reshape(nssd, nss_per, dvg.shape[-1])
    gtraces = [dvg[i, 0] for i in [2, -1]]
    straces = [dvs[0, i, 0] for i in [2, -1]]
    gtraces = [gt[gt <= bound] for gt in gtraces]
    ssi, xinit_ss, integrated, dvgs, dvss = [], [], [], [], []
    for i, (g, s) in enumerate(zip(gtraces, straces)):
        xinit_ss.append(Tg[0] - Ts[i])
        ssi.append(g[:xinit_ss[i]])
        ss = s[:Ts[i]]
        s = np.append(g[:xinit_ss[i]], ss[ss >= 0])
        ixmin = np.min([len(g), len(s)])
        dvgs.append(g[:ixmin])
        dvss.append(s[:ixmin])
    nframes = [len(gt) for gt in dvgs]
    x = params['tr'][0] * 1000 + [np.arange(nf) for nf in nframes]
    sim.__update_steps__(dt=.005)
    return [x, dvgs, dvss, xinit_ss, ssi, np.max(nframes)]

def build_decision_axis(onset, bound, ssd=np.array([300, 400]), tb=650):
    # init figure, axes, properties
    f, axes = plt.subplots(len(ssd), 1, figsize=(7, 7), dpi=300)
    #f.subplots_adjust(wspace=.05, top=.1, bottom=.1)
    f.subplots_adjust(hspace=.05, top=.99, bottom=.09)
    w = tb + 40
    h = bound
    start = onset - 80
    for i, ax in enumerate(axes):
        plt.setp(ax, xlim=(start - 1, w + 1), ylim=(0 - (.01 * h), h + (.01 * h)))
        ax.vlines(x=ssd[i], ymin=0, ymax=h, color="#e74c3c", lw=2.5, alpha=.5)
        ax.hlines(y=h, xmin=start, xmax=w, color='k')
        ax.hlines(y=0, xmin=start, xmax=w, color='k')
        ax.vlines(x=tb, ymin=0, ymax=h, color='#2043B0', lw=2.5, linestyle='-', alpha=.5)
        ax.vlines(x=start + 2, ymin=0, ymax=h, color='k')
        ax.text(ssd[i] + 10, h * .88, str(ssd[i]) + 'ms', fontsize=19)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(top=True, right=True, bottom=True, left=True)
    return f, axes

def re_animate_multiax(i, x, gtraces, glines, straces, slines, params, xi, yi):
    gcolor = '#27ae60'
    scolor = '#e74c3c'
    for n in range(len(x)):
        ex, gt, gl, st, sl, ix, iy = [xx[n] for xx in [x, gtraces, glines, straces, slines, xi, yi]]
        try:
            gl.set_data(ex[:i + 1], gt[:i + 1])
            gl.set_color(gcolor)
            sl.set_data(ex[ix:i + 1], st[ix:i + 1])
            sl.set_color(scolor)
        except Exception:
            return sl, gl
        #f.savefig('animation_frames/movie/img' + str(i) +'.png', dpi=300)
    return sl, gl

def anim_to_html(anim):
    from tempfile import NamedTemporaryFile
    VIDEO_TAG = """<video controls>
             <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
             Your browser does not support the video tag.
            </video>"""
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)
