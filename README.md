# Harapanahalli_CDC2024

This is the code accompanying the CDC 2024 paper titled "Efficient Reachable Sets on Lie Groups Using Lie Algebra Monotonicity and Tangent Intervals".

## Setup and installation

Clone the repository and its dependency `npinterval`, our interval arithmetic package for `numpy`.
```shell
git clone --recurse-submodules https://github.com/gtfactslab/Harapanahalli_CDC2024.git
```
We recommend installing into a fresh `conda` environment to avoid any issues with other packages.
```shell
conda create -n liereach python=3.11
conda activate liereach
```
Install the required packages.
```shell
pip install -r requirements.txt
```

## Torus 

Change into the `Tn` directory, and run `reach.py`.
```shell
cd Tn
python reach.py
```
We assume you have `ffmpeg` installed with the `h264` codec to generate the `.mp4` files.

## SO(3)

Change into the `SO3` directory.
```shell
cd SO3
```

To generate Figure 2 from the paper, run `compare.py`.
```shell
python compare.py
```

To run the example, run `reach.py`.
```shell
python reach.py
```
We assume you have `ffmpeg` installed with the `h264` codec to generate the `.mp4` files.
