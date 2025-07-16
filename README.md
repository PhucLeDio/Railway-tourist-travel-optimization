## Program Objective Overview

The primary goal of this program is to find an optimal travel route among a set of locations (referred to as "points" or "nodes"), starting and ending at a specific railway station. The program doesn't just seek a single "best" route but aims to discover a set of Pareto optimal solutions. Pareto optimal routes are those where you cannot improve one objective (e.g., reduce time) without worsening another (e.g., increase cost).
After finding a good route using ACO, the program refines that route using 2-Opt to further optimize the cost for a smaller, selected set of Points of Interest (POIs).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage

```bash
venv/Scripts/activate
```
```bash
python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
