# Running

suggested to use UV, https://github.com/astral-sh/uv

```bash
uv sync
uv run python main.py
```

try to make sure gpu is actually used, if main viz is slow

```
uv run python
import vispy
print(vispy.sys_info())
````

if not, you should try to install opengl drivers libs whatever, maybe test with smaller vispy example

## OSX

on osx maybe uncomment quartz package in pyproject.toml for better performance


## Windows

1. install uv
2. edit pyproject.toml and <b>un</b>-comment last line `constraint-dependenc...`
3. run:

```
uv sync
uv run python main.py
```


