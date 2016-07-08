#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd import theta
from IPython.display import HTML

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
