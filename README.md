# Proof Of Concept/Test for multilayer visualization

in experimental alpha state :)


# Running

suggested to use UV, https://github.com/astral-sh/uv

```bash
uv sync
uv run python main.py
```

## Linux

```bash
uv sync
uv run python main.py
```

if no opengl backend or errors try this to debug (maybe try to get glxgears or vispy minimal example running first):

```
uv run python
import vispy
print(vispy.sys_info())
````

if not, you should try to install opengl drivers libs whatever, maybe test with smaller vispy example

## OSX

```bash
uv sync
uv run python main.py
```

maybe uncomment quartz package in pyproject.toml for better performance


## Windows

1. install uv
2. edit pyproject.toml and <b>un</b>-comment last line `constraint-dependenc...`
3. run:

```bash
uv sync
uv run python main.py
```

# 3d Visualization Tipps

## Try different layout algorithms

* click in 3d place to activate keyboard shortcuts
* press `z` for a top view to easily see the aggregated view of layers
* select algorithm in dropdown at the loader panel at the top and press load button
* there are some custom algorithms starting with "cluster_" that primarily focus on keeping clusters together

## Mouse & Keyboard Interaction

* rotate: mouse drag
* panning: shift + mouse-drag
* zoom/change distance: scrollwheel or two finger drag on mac trackpad, or shift + right click

Click into 3D Viz to activate keyboard inputs.

* `z` view from top a long z axis
* `x` view along x axis
* `y` view along x axis
* `r` reset camera
* `c` center camera
* `w` and `s`, `a` and `d`, `q` and `e`, rotate slightly around axis

## Aggregated Layer View

* click into 3d view to enable keyboard shortcuts
* press `z`

## Inspect interlayer-edges of node

* disable `Intralayer Edges` in `Display Options`
* enable `Show Node Labels`
* click into 3d view to enable keyboard shortcuts
* press `z`
* press `s` one time
* zoom in very very close

now you should see vertical lines for each Interlayer Edge (arranged in circle around node for better distinguishing (no overlapping))

## Quick Node based Layer statistics

* disable `Intralayer Edges` and `Show Nodes`
* enable `Node Labels` and `Show Inter Stats Bars`
* click in 3D view to enable keyboard shortcuts
* press `z`

now you should see in-place bar charts with
* number of active layer for node
* number of Interlayer edges
(this corresponds to the numbers in the node label)
