The project consists of both code and sample data

## Project Structure 
project-root/
├── Code/
│ ├── DFS_BFS_code/
│ │ ├── DFS_BFS_Traversal_example.py
│ │ └── DFS_cycle_detection_example.py
│ ├── HOMcode/
│ │ ├── BuildHON.py
│ │ ├── BuildHONPlus.py
│ │ ├── GrowHON.py
│ │ └── main.py
│ ├── MetaPathcode/
│ │ ├── table1_code.py
│ │ └── table2_code.py
│ └── RandomWalkcode/
│ ├── detect.py
│ └── random_walk.py
└── Sample_data/
├── DFS_BFS_Sample_data/
│ └── Sample_data.csv
├── HOM_Sample_data/
│ └── trajectories_Sample_data.txt
├── MetaPath_Sample/
│ ├── InvestmentBehavior_Sample_data.txt
│ └── trajectories_Sample_data.txt
└── RandomWalk_Sample_data/
└── Sample_data.csv

## Module Description

### DFS_BFS_code
- `DFS_BFS_Traversal_example.py`: Implementation of DFS and BFS traversal algorithms
- `DFS_cycle_detection_example.py`: Implementation of cycle detection using DFS

### HOMcode
Higher-Order Network construction related code:
- `BuildHON.py`: Basic HON construction
- `BuildHONPlus.py`: Enhanced HON construction
- `GrowHON.py`: HON growth algorithm
- `main.py`: Main program entry

### MetaPathcode
Meta-path analysis related code:
- `table1_code.py`: Meta-path statistics analysis
- `table2_code.py`: Investment network meta-path analysis

### RandomWalkcode
Random walk related algorithms:
- `detect.py`: program entry
- `random_walk.py`: Random walk implementation

## Sample Data Description

### Sample_data
Contains sample data for each module:
- DFS_BFS sample data
- Higher-order network (HOM) trajectory data
- Investment behavior data for meta-path analysis
- Random walk sample data

## Requirements
- Python 3.x
- NetworkX
- Pandas
- NumPy
