# Stabilizer States
[![pypi](https://img.shields.io/pypi/v/stabilizer-states.svg)](https://pypi.org/project/stabilizer-states/)

A supporting, optional package to [stabilizer-toolkit](https://github.com/amirebrahimi/stabilizer-toolkit), which 
provides the following stabilizer state datasets:
* 1 qubit stabilizer states (all: 6)
* 2 qubit stabilizer states (all: 60)
* 3 qubit stabilizer states (all: 1080)
* 4 qubit stabilizer states (all: 36720)
* 5 qubit stabilizer states (all: 2423520, ternary: 146880)
* 6 qubit stabilizer states (ternary: 9694080)

## Usage

```python
from stabilizer_states import StabilizerStates
S1 = StabilizerStates(1)
print(S1.shape)
```

It's worth noting that if you are only using the real or ternary form of the states, then in some cases the dataset will
load quicker (e.g. 5 and 6 qubit datasets):
```python
from stabilizer_states import StabilizerStates
S5 = StabilizerStates(5, "ternary")
print(S5.shape)
```

## Installation
The package is available on [pypi](https://pypi.org/project/stabilizer-states/) and installable via `pip`:
```shell
pip install stabilizer-states 
```

## Development
If you'd like to work with a development copy, then clone this repository and install via
[`poetry`](https://python-poetry.org/docs/#installation):
```shell
poetry install -vvv --with dev
```