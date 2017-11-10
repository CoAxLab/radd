#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd import theta
from IPython.display import HTML
from matplotlib import animation

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


def render_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def animate_dpm(model):
    """ to render animation within a notebook :
        vis.render_animation(vis.animated_dpm_example(MODEL))
    """
    model.set_fitparams(dt=.001)
    params = deepcopy(model.inits)
    bound=theta.scalarize_params(params)['a']
    # generate reactive model simulations
    x, goTraces, brakeTraces, xi, yi, nframes = gen_re_traces(model)
    f, axes = build_decision_axis(onset=x[0][0], bound=bound)
    # axes line object for "go" process
    goLine = [axes[i].plot([], [], linewidth=1.5)[0] for i, n in enumerate(gtraces)]
    # axes line object for "brake" process
    brakeLine = [axes[i].plot([xi[i]], [yi[i]], linewidth=1.5)[0] for i, n in enumerate(straces)]
    f_args = (x, goTraces, goLine, brakeTraces, brakeLine, params, xi, yi)
    anim=animation.FuncAnimation(f, re_animate_multiax, fargs=f_args, frames=nframes, interval=1, blit=True)
    return anim


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


def gen_re_traces(model):
    params = deepcopy(model.inits)
    dvg, dvs, rt, ssrt = model.sim.simulate_traces(params)
    bound = params['a']
    tr = params['tr']
    goOn = model.sim.gOnset[0]
    ssOn = model.sim.ssOnset.squeeze()
    ssOn = [ssOn[2], ssOn[-1]]
    xinit_ss = [ssOn_i-goOn for ssOn_i in ssOn]
    rtSteps = [int(rt[0, i]*1000)+1 for i in [0, 1]]
    ssrtSteps = [int(ssrt[0, 2, 0]*1000)+1, int(ssrt[0, -1, 0]*1000)+1]
    gtraces = [dvg[0, 0, goOn:rtSteps[0]], dvg[0, 1, goOn:rtSteps[-1]]]
    straces = [dvs[0, 2, 0, ssOn[0]:ssrtSteps[0]], dvs[0, -1, 0, ssOn[1]:ssrtSteps[1]]]
    # gtraces = [gt[gt <= bound] for gt in gtraces]
    ssi, integrated, dvgs, dvss = [], [], [], []
    for i, (g, s) in enumerate(zip(gtraces, straces)):
        ssi.append(g[:xinit_ss[i]])
        ss = s[:ssOn[i]]
        s = np.append(g[:xinit_ss[i]], ss[ss >= 0])
        ixmin = np.min([len(g), len(s)])
        dvgs.append(g[:ixmin])
        dvss.append(s[:ixmin])
    nframes = [len(gt) for gt in dvgs]
    x = [np.arange(goOn, goOn + nf) for nf in nframes]
    return [x, dvgs, dvss, xinit_ss, ssi, np.max(nframes)]


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



def gen_ddm_traces(model, ntrials=10):
    params = deepcopy(model.inits)
    dvg, dvs, rt, ssrt = model.sim.simulate_traces(params)
    bound = params['a']
    tr = params['tr']
    goOn = model.sim.gOnset[0]
    trials = range(ntrials)
    rtSteps = [int(rt[0, t]*1000)+1 for t in trials]
    gtraces = [dvg[0, t, goOn:rtSteps[t]] for t in trials]
    nframes = [len(gt) for gt in gtraces]
    x = [np.arange(goOn, goOn + nf) for nf in nframes]

    return [x, gtraces, np.max(nframes)]


def animate_ddm_func(i, x, gtrace, glines, line_i):
    gcolor = '#4168B7'
    # for n in range(len(x)):
    # ex, gt, gl = [xx[n] for xx in [x, gtraces, glines]]
    try:
        line_index = line_i[i]
        gl = glines[line_index]
        xx = x[line_index]
        # gt = gtraces[line_index]
        if x==200:
            gl.set_data(x[i:i + 1], gtraces[i:i + 1])
        else:
            gl.set_data(x[:i + 1], gtraces[:i + 1])
        gl.set_color(gcolor)
    except Exception:
        return gl
        #f.savefig('animation_frames/movie/img' + str(i) +'.png', dpi=300)
    return gl
