# Understanding Imbalanced Forgetting in Class-Incremental Learning with Class-Balanced Rehearsal

> Alberto Tamajo, Srinandan Dasmahapatra and Rahman Attar

Official repository for the paper [Understanding Imbalanced Forgetting in Class-Incremental Learning with Class-Balanced Rehearsal](https://arxiv.org/abs/2605.14785) (under review).

## Abstract
Neural networks suffer from *catastrophic forgetting* in class-incremental learning (CIL) settings. Rehearsal—replaying a subset of past samples—is a well-established mitigation strategy. However, recent results suggest that, despite balanced rehearsal allocation, some classes are forgotten substantially more than others. Despite its relevance, this *imbalanced forgetting* phenomenon remains underexplored. This work shows that imbalanced forgetting arises systematically and severely in CIL with class-balanced rehearsal, and investigates it extensively. Specifically, we construct, from a step-wise optimization analysis, three last-layer coefficients that capture different sources of interference affecting each past class during an incremental step. We then show that the coefficients capturing self-induced bias interference (SIC) and interference from new classes (NIC) exhibit robust associations with class-wise forgetting, whereas the coefficient capturing bias interference from other past classes (CIC) is only weakly correlated. Moreover, combining all three generally yields a better account of the observed ranking of past classes by their forgetting, suggesting that they capture relevant, potentially complementary properties of imbalanced forgetting. Ultimately, we find that the higher SIC values of the most forgotten classes reflect insufficient beneficial support from their rehearsal last-layer gradients to counterbalance their larger original last-layer gradient magnitudes, with controlled experiments suggesting that larger NIC values may contribute to these larger norms. Based on these findings, we hypothesize that discrepancies in NIC across past classes may contribute to the emergence of imbalanced forgetting, while insufficient rehearsal support relative to class-specific demands, as captured by SIC, may contribute to its persistence.

## Repository Structure

```text
.
├── Analysis/          # Scripts for analyzing data extracted from completed experiments
├── Experiments/       # Scripts for running the experiments presented in the paper
├── Extractions/       # Scripts for extracting data from completed experiments
├── benchmarks/        # CIFAR-100, TinyImageNet, and ImageNet benchmarks adapted to class-incremental learning
├── convnets/          # ResNet backbone implementations
├── metrics/           # Metrics tracked during continual learning
├── models/            # Building blocks for continual learning architectures
├── training/          # Continual learning training loops
├── LICENSE            # Repository license
├── README.md          # Readme file
└── environment.yml    # Conda environment file for reproducing the project environment
```

The `benchmarks`, `metrics`, `models`, and `training` modules build on the corresponding modules from the [Avalanche](https://avalanche.continualai.org/) framework. We thank the authors for their contributions to the continual learning community.

## Reproducibility
This repository provides two different reproducibility workflows.
### Full Reproducibility
To fully reproduce the results presented in the paper, follow the instructions below in the given order:
1. We provide an `environment.yml` file containing a list of the necessary dependencies. Use it to reproduce the same environment in your machine.
2. Run all the scripts inside the `Experiments` folder.
3. Run all the scripts inside the `Extractions` folder.
	- These scripts produce `.pkl` files containing the extracted data. Move these files into the respective subfolders within the `Analysis` folder.
4. Use the analysis scripts located in the subfolders of the `Analysis` folder to analyze the extracted data.
### Lightweight Reproducibility
Alternatively, since full reproduction is time-consuming, you can skip Steps 1–3 by downloading the ZIP archives containing the .pkl files generated during our empirical investigation from our Zenodo repository. Extract the archives, move the files to their respective subfolders within the Analysis folder, and proceed directly to Step 4.

## Citation
If you find our work useful in your research, please consider citing:

	@article{tamajo2026understanding,
  		title={Understanding Imbalanced Forgetting in Class-Incremental Learning with Class-Balanced Rehearsal},
  		author={Tamajo, Alberto and Dasmahapatra, Srinandan and Attar, Rahman},
  		journal={arXiv preprint arXiv:2605.14785},
  		year={2026}
  	}

	
## License
Our code is released under MIT License (see LICENSE file for details).

