# Galaxy N-body simulation (Futhark + Python)

A rotating disk galaxy simulated with direct N-body gravity. Futhark runs the
O(n^2) force computation each timestep; morloc drives the timeline, threading the
`(position, velocity)` state through each step and rendering as it goes; Python
seeds the galaxy, draws each frame, and encodes the video.

- **Futhark** (`kernels.fut`): pairwise gravity + leapfrog integration; `(pos, vel)`
  in and out as a tuple.
- **Python** (`galaxy.py`): seed a disk around a heavy central mass (data prep),
  render frames, encode an MP4 (visualization).
- **morloc** (`main.loc`): the effectful timestep loop -- a `<IO>` recursion that
  renders the current positions, advances one Futhark step, and recurses.

## Run

```bash
morloc make -o nexus main.loc
./nexus simulate 400 60 out     # 400 bodies, 60 frames, written under out/
# -> out/galaxy.mp4
```

Requires `imageio`/ffmpeg for the video stage.
