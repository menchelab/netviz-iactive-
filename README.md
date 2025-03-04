# Proof Of Concept/test for multilayer network visualisation -> filtering -> analysis

in experimental state :)


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


