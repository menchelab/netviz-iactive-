# Running

suggested to use UV, https://github.com/astral-sh/uv

```bash
uv sync
uv run python main.py
```

try to make sure gpu is actually used

```
uv run python
import vispy
print(vispy.sys_info())
````

if not, you should try to install opengl drivers libs whatever, maybe test with smaller vispy example

on osx maybe uncomment quartz package in pyproject.toml

(vispy uses opengl, this is tested on osx, win/linux you might need other sys packages, i think on osx i installed glew (brew install glew))