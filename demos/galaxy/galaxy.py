# Data prep + visualization for the galaxy N-body demo. Python seeds a rotating
# disk around a heavy central mass (deterministic), renders each frame the morloc
# driver hands it, and finally encodes the frames into a video. The physics loop
# lives in Futhark; morloc owns the timeline.
#
# initPos / initVel / initMass each rebuild the same seeded galaxy and return one
# component, so the three stay consistent without crossing a tuple.

import os


def _galaxy(n):
    import numpy as np
    rng = np.random.default_rng(7)
    n = int(n)
    pos = np.zeros((n, 3), dtype=np.float32)
    vel = np.zeros((n, 3), dtype=np.float32)
    mass = np.full(n, 1.0, dtype=np.float32)
    # central body
    mass[0] = 4000.0
    # disk particles on near-circular orbits
    r = np.sqrt(rng.uniform(0.02, 1.0, size=n - 1)) * 8.0
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n - 1)
    pos[1:, 0] = r * np.cos(theta)
    pos[1:, 1] = r * np.sin(theta)
    pos[1:, 2] = rng.normal(0.0, 0.05, size=n - 1)
    vcirc = np.sqrt(mass[0] / (r + 0.1))
    vel[1:, 0] = -vcirc * np.sin(theta)
    vel[1:, 1] = vcirc * np.cos(theta)
    return pos, vel, mass


def init_pos(n):
    return _galaxy(n)[0]


def init_vel(n):
    return _galaxy(n)[1]


def init_mass(n):
    return _galaxy(n)[2]


def render_frame(pos, outdir, frame):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    p = np.asarray(pos)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(p[1:, 0], p[1:, 1], s=2, c="#9fc5ff", alpha=0.7)
    ax.scatter(p[:1, 0], p[:1, 1], s=40, c="#ffd27f")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(outdir, "frame_%04d.png" % int(frame)),
                dpi=80, facecolor="black")
    plt.close(fig)
    return None


#  def encode_video(outdir):
#      import imageio.v2 as imageio
#      frames = sorted(f for f in os.listdir(outdir) if f.startswith("frame_"))
#      out = os.path.join(outdir, "galaxy.mp4")
#      with imageio.get_writer(out, fps=20) as w:
#          for f in frames:
#              w.append_data(imageio.imread(os.path.join(outdir, f)))
#      return out

def encode_video_gif(outdir):
    import imageio.v2 as imageio
    frames = sorted(f for f in os.listdir(outdir) if f.startswith("frame_"))
    out = os.path.join(outdir, "galaxy.gif")
    with imageio.get_writer(out, fps=20, loop=0) as w:
        for f in frames:
            w.append_data(imageio.imread(os.path.join(outdir, f)))
    return out
