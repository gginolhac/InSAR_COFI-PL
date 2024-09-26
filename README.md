# InSAR COFI-PL

This repertory contains the code for the paper:
> Covariance Fitting Interferometric Phase Linking: Modular Framework and Optimization Algorithms, submitted to IEEE TGRS

Preprint available at: https://arxiv.org/abs/2403.08646

## Environnement

Use the provided conda environnement:
```console
conda env create -f environnement.yml
conda activate InSAR
```

## Running experiments

Computations of cost functions w.r.t. iterations:
```
script_cost_functions
```
Computations of MSE to compare distances:
```
script_mse_compar_distance
```
Computations of MSE to compare estimators:
```
script_mse_compar_estimators
```
Computations of MSE to compare regularizations:
```
script_mse_compar_regularization
```

## Citation

If you use any code in this repository please cite us using:
>
@misc{vu2024covariancefittinginterferometricphase,
      title={Covariance Fitting Interferometric Phase Linking: Modular Framework and Optimization Algorithms}, 
      author={Phan Viet Hoa Vu and Arnaud Breloy and Frédéric Brigui and Yajing Yan and Guillaume Ginolhac},
      year={2024},
      eprint={2403.08646},
      archivePrefix={arXiv},
      primaryClass={stat.AP},
      url={https://arxiv.org/abs/2403.08646}, 
}

## Authors

* Guillaume, Ginolhac
* 

