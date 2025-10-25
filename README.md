# WSCAN++

## Requirements
- Python
- Package needed:
  - networkx  
  - psutil  
  - matplotlib  
  - numpy  
  - scikit-learn 

## Dataset Structure
- ../dataset/real/<network_name>/network.dat   # weighted edge list (u v weight)
- ../dataset/real/<network_name>/labels.dat    # ground truth labels

## Arguments description
- eps : similarity threshold
- mu : core threshold
- gamma : balance factor

- similarity : how to compute similarity (ours : WSCAN++)
- network : path to network dataset ; only insert network name

- dataclass : real data or synthetic data (for dataset path)
- exp_mode : what is the experiment for? (ex. effectiveness or time ..)

- delta_p : percentage for delta (delta is the parameter for perturbing edge weight)
- edge_p : percentage of edges to perturb

- use_parallel : whether to use parallel mode in core computing
- process_num : how many processes for parallel

## Usage example
```
python main.py --network karate --eps 0.25 --mu 2 --gamma 0.5 --similarity WSCAN++ --dataclass real --exp_mode effectiveness
--use_parallel True --process_num 32
```