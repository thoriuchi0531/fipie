# fipie 

**fipie** is a simple portfolio optimiser which allows you to compute asset weights with algorithms from Markowitz's
mean-variance optimisation to more modern methodologies!

## Installation

```
pip install git+https://github.com/thoriuchi0531/fipie.git
```

## Example

The simplest example is to compute equal nominal weights without clusters.

```python
from fipie import Portfolio, EqualWeight
from fipie.data import load_example_data

# Create a portfolio instance
price = load_example_data()
# Use weekly returns to calculate weights -- but this does not matter for equal weighting.
ret = price.asfreq('w', method='pad').pct_change().dropna()
portfolio = Portfolio(ret)

# Compute the latest weight
portfolio.weight_latest(EqualWeight())
```
As expected, each instrument gets a weight of 1/7.
```python
node_id
SPY    0.142857
IWM    0.142857
QQQ    0.142857
MDY    0.142857
TLT    0.142857
GLD    0.142857
USO    0.142857
Name: weight, dtype: float64
```

The portfolio above actually contains similar ETFs (e.g., SPY and IWM which are both US equities). 
In this scenario, it's more desirable to group similar instruments into clusters first and then compute weights.

```python
from fipie import CorrMatrixDistance

# Compute the latest weight with the clustering algorithm
cluster_algo = CorrMatrixDistance(max_clusters=3)
portfolio.weight_latest(EqualWeight(), cluster_algo)
```

With clusters, we can see different weights -- SPY and IWM are still equally weighted, but much smaller weights. 
On the other hand, TLT has got a much larger weight.

```python
node_id
SPY    0.055556
IWM    0.055556
QQQ    0.166667
MDY    0.055556
TLT    0.333333
GLD    0.166667
USO    0.166667
Name: weight, dtype: float64
```

This is because the instruments are grouped as follows.

```python
portfolio.create_tree(cluster_algo).show()
```
TLT has its own group, while equity ETFs are grouped into a cluster. 
There is another cluster for commodities (GLD and USO), resulting in weights somewhat between equities and bonds.
```python
Node(root)
    Node(cluster_0)
        Node(cluster_1)
            Node(SPY)
            Node(IWM)
            Node(MDY)
        Node(QQQ)
    Node(cluster_2)
        Node(GLD)
        Node(USO)
    Node(TLT)
```

# More details

```{toctree}
:maxdepth: 2
:caption: API reference
:glob:

api/*
```
