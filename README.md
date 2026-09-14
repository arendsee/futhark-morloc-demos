# Morloc+Futhark examples

## Morloc Install

```
# download the Morloc Installation Manager (mim) to ./local/bin/
curl -fsSL https://raw.githubusercontent.com/morloc-project/morloc-manager/main/scripts/install.sh | sh

# bulid the environment
mim new --lang futhark

# drop into a shell
mim shell
```

From within this shell, all the examples in this repo can be built and run with
`make` commands from each project directory.

These have all been tested on Morloc v0.105.0

## galaxy: N-body simulations

<img src="./assets/galaxy.gif" width="600">

## sobel: N-body simulations

<img src="./assets/het.png" alt="Sobel transformed image" style="width:66%">

## k-means clustering

<img src="./assets/kmeans.png" width="600">
