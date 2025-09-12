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

## Usage
```
python main.py --network karate --eps 0.25 --mu 2 --gamma 0.5 --similarity Gen --gt True
```