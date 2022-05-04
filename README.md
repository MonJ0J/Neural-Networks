# Part 4
Implementation of a simple neural network


## Dependencies:

* **os**
* **random**
* **numpy**

## How to use:

* On the CLI: navigate to the Lab D folder that contains input data files, as well as juanjunco-anaverulidze-labD.py and type ```python3 juanjunco-anaverulidze-labD.py -i file-name -n num-inputs -p num-hiddens -o num-outputs -e epochs -l learning-rate```. Make sure to replace "file-name", "num-inputs", "num-hiddens", "num-outputs", "epochs", "learning-rate", with the according values. File name and learning rate are string and float in that order, while all the other inputs are ints.

### example command:
```
AnaVerulidze_JuanJunco_lab_c.py -i xor.txt -n 2 -p 2 -o 1 -e 100 -l 0.3
```
### example output:
```
raw_error [[ 0.5        -0.11207102 -0.66490681  0.79971607]]
output_weights_gradient [[0.0991373  0.08249231]]
output_bias_gradient [[0.13068456]]
blame_array [[-0.34261912  0.07679535  0.45561957 -0.54799603]
 [ 1.03487948 -0.23196    -1.37619684  1.6552195 ]]
hidden_layer_outputs [[0.5        0.77730638 0.74276201 0.90973567]
 [0.5        0.57885522 0.7897941  0.83777434]]
hidden_outputs_squared [[0.25       0.60420521 0.55169541 0.82761899]
 [0.25       0.33507336 0.62377472 0.70186585]]
propagated_error [[-0.25696434  0.0303952   0.20425635 -0.09446411]
 [ 0.77615961 -0.15423638 -0.51776004  0.49347746]]
hidden_weights_gradient [[ 0.02744806 -0.01601723]
 [-0.00607065  0.08481027]]
hidden_bias_gradient [[-0.02919423]
 [ 0.14941016]]
LOSS 1.1356396636809456

raw_error [[ 0.5        -0.04071842 -0.9506701   0.55004867]]
output_weights_gradient [[ 0.08234148 -0.09463846]]
output_bias_gradient [[0.01466504]]
blame_array [[-1.47931841  0.12047101  2.81268757 -1.62739425]
 [ 1.57975209 -0.12865001 -3.00364616  1.73788108]]
hidden_layer_outputs [[3.22212049e-01 9.84263866e-01 4.77568089e-03 3.87016059e-01]
 [1.57985980e-02 1.24088139e-05 4.06812072e-01 5.29873786e-04]]
hidden_outputs_squared [[1.03820605e-01 9.68775358e-01 2.28071279e-05 1.49781430e-01]
 [2.49595700e-04 1.53978661e-10 1.65496062e-01 2.80766229e-07]]
propagated_error [[-1.32573468  0.00376166  2.81262342 -1.38364082]
 [ 1.57935779 -0.12865001 -2.50655455  1.73788059]]
hidden_weights_gradient [[ 0.35724565 -0.34496979]
 [-0.19216849  0.40230765]]
hidden_bias_gradient [[0.0267524 ]
 [0.17050846]]
{'weight_hidden': array([[-4.70298068,  4.98304999],
       [ 3.81239998, -7.28587408]]), 'output_biases': array([[-1.41442105]]), 'weight_output': array([[-2.98333926,  3.18789571]]), 'hidden_biases': array([[-0.75165034],
       [-4.18306188]])}

```
