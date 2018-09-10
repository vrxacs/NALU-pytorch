# Neural Arithmetic Logic Units

[WIP]

This is a PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom*.

<p align="center">
 <img src="./imgs/arch.png" alt="Drawing", width=60%>
</p>

## API

```python
from models import *

# single layer modules
NeuralAccumulatorCell(in_dim, out_dim)
NeuralArithmeticLogicUnitCell(in_dim, out_dim)

# stacked layers
NAC(num_layers, in_dim, hidden_dim, out_dim)
NALU(num_layers, in_dim, hidden_dim, out_dim)
```

## Experiments

To reproduce "Numerical Extrapolation Failures in Neural Networks" (Section 1.1), run:

```python
python failures.py
```

This should generate the following plot:

<p align="center">
 <img src="./imgs/extrapolation.png" alt="Drawing", width=60%>
</p>

To reproduce "Simple Function Learning Tasks" (Section 4.1), run:

```python
python function_learning.py
```
This should generate a text file called `interpolation.txt` with the following results. (Currently only supports interpolation, I'm working on the rest)

|         | Relu6    | None     | NAC      | NALU   | fauxNALU   |
|---------|----------|----------|----------|--------|------------|
| a + b   | 4.837    | 0.137    | 0.049    | 0.230  | 0.326      |
| a - b   | 92.562   | 1.901    | 0.462    | 62.604 | 1.214 	 |
| a * b   | 49.224   | 0.408    | 89.212   | 0.380  | 0.511      |
| a / b   | 104.827  | 6.532    | 7373.894 | 0.186  | 6.039 	 |
| a ^ 2   | 50.320   | 0.492    | 89.430   | 0.711  | 0.421      |
| sqrt(a) | 13.426   | 4.493    | 2385.364 | 0.326  | 8.255      |
