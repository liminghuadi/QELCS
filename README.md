# QELCS
Quality‑aware Elite Learning‑driven Complementary Search for Feature Selection in Microarray‑based Cancer Classification(QELCS). There are some caveats to running this code: 1、To illustrate, only a subset of the complete dataset is provided here as an example. The entire dataset can be obtained from http://featureselection.asu.edu. 2、After downloading the raw dataset from a public database, preprocessing steps should be performed to convert the dataset into feat and label format for algorithm readability.

# QELCS for High-Dimensional Feature Selection

QELCS selects feature subsets for supervised classification by combining stability-adjusted feature ranking with particle swarm optimization. This MATLAB repository provides the Case A implementation, the experimental protocol, and results for 21 benchmark datasets.

## Getting started
The recorded verification environment is MATLAB R2022b with Statistics and Machine Learning Toolbox 12.4. Set the repository root as the MATLAB current folder and run a single experiment on Colon.
```matlab
summary = main('quick', 'Colon');
disp(summary(:, {'Dataset', 'TestAccMean', 'FeatureMean'}))
```
This command runs one complete nested cross-validation repeat with a reduced search budget. The summary and fold-level results are written to `results/quick/`.

## Method
For a subset S drawn from $d$ input features, QELCS minimizes
Fitness=α∙error+(1-α)∙selected/all
where error is the fraction of misclassified samples pooled across the inner validation folds. Predictions use a Euclidean 1-nearest-neighbor classifier on the original feature scale.

1. Candidate selection. Each inner training fold yields two scores per feature: symmetrical uncertainty with the class label and the mean absolute correlation with one-versus-rest class indicators. The algorithm penalizes variation across folds, rescales each score, and combines them with weights of 0.70 and 0.30, respectively. The top 10% of features form the candidate set. All particles share a high-ranked core at initialization, with successively larger portions of the ranked list available to later particles.
2. Swarm search. At each iteration, the algorithm assigns particle roles using the classification error and feature proportion of each personal best. Particles with relatively dense, accurate subsets follow a sparsity-oriented leader. The other particles follow an accuracy-oriented leader. The corresponding selection thresholds converge from 0.60 and 0.40 to 0.50.
3. Elite archive. The archive retains distinct subsets within a specified tolerance of the best fitness. Pooled error and variability in fold-level error guide the choice of leaders. Personal and global bests are updated whenever the objective Fitness strictly improves.

## Experimental protocol
Each repeat starts by generating all outer and inner partitions from its random seed. Within each outer training set, QELCS ranks features and searches for a subset using five-fold inner cross-validation. The classifier then predicts the held-out outer fold using the selected subset. Pooling the ten outer folds yields one held-out prediction per sample per repeat.
| Configuration | Default datasets | Particles | Iterations | Repeats | Outer / inner folds |
| --- | --- | ---: | ---: | ---: | ---: |
| 'paper' | 21 benchmarks | 100 | 100 | 5 | 10 / 5 |
| 'quick' | Colon | 10 | 10 | 1 | 10 / 5 |

The five formal repeats use seeds 43 through 47 with the MATLAB `twister` generator. Search parameters are defined in [parameters.m](parameters.m).

### Running the benchmarks
```matlab
% Run two benchmarks with the full search budget and five repeats.
outDir = fullfile(pwd, 'results', 'paper_subset');
summary = main('paper', ["Colon", "nci9"], [], outDir);

% Run all 21 benchmarks in a separate output directory.
outDir = fullfile(pwd, 'results', 'paper_full');
summary = main('paper', [], [], outDir);
```
Calling `main` without arguments runs the full benchmark suite in `results/paper/`. Reusing an output directory replaces results for datasets with the same name and rewrites the summary for the current invocation.

### Input data

The [DataSet](DataSet/) directory contains 16 MAT files. The default benchmark list in [main.m](main.m) selects 16 of them. Each file must contain a finite numeric matrix `feat` (samples by features) and a vector 'label' with one nonmissing class label per sample. A dataset can be selected by its filename stem, as in 'main('quick', 'Brain_Tumor2')'.

## Results

The repository includes the five-repeat summary for all 21 benchmarks. Accuracy values in the CSV files are expressed as proportions. [Benchmark summary](results/paper/summary.csv)

| Per-repeat metric | Definition |
| --- | --- |
|'TestAcc' | Fraction of correct predictions across all held-out outer-fold samples. |
|'TrainCV' | Mean of the ten inner cross-validation accuracies, one for the subset selected in each outer fold. |
|'Gbest' | Mean best objective value across the ten outer folds. Lower values indicate better fitness. |
|'F' | Mean number of selected features across the ten outer folds. |

Each run produces a per-dataset table ('<dataset>_runs.csv') with one row per repeat and a MAT file ('<dataset>_result.mat') containing the selected features, convergence curves, partitions, predictions, and parameters. To inspect the first repeat and outer fold from the quick-start example, use

```matlab
r = load(fullfile('results', 'quick', 'Colon_result.mat'), 'datasetResult');
featureIdx = r.datasetResult.selectedFeatures{1, 1};
fitnessCurve = r.datasetResult.convergenceCurves{1, 1};
```

## Reproducibility

The archived verification report documents full reruns of Colon and nci9 against the saved Case A reference. The values below are means over five repeats. [Verification report](results/verification/verification.csv)

| Dataset | Outer test accuracy | Mean selected features | Reference comparison |
| --- | ---: | ---: | --- |
| Colon | 77.74% | 22.08 | Exact match |
| nci9 | 61.00% | 147.64 | Exact match |

To repeat the recorded comparison, run

```matlab
addpath('verification')
report = verify_reproduction();
disp(report)
```

The verifier uses 'isequaln' to compare seeds, partition indices, selected feature indices, convergence histories, fold metrics, and per-repeat metrics. Runtime is excluded. Use 'verify_reproduction('Colon')`'for one dataset or 'verify_reproduction('all')' for every dataset in the reference file. Reports are written to 'results/verification/'.

## Source files

| File | Responsibility |
| --- | --- |
| [main.m](main.m) | Dataset loading, nested cross-validation, and result export |
| [parameters.m](parameters.m) | Case A search parameters and experiment configurations |
| [gQELCS.m](gQELCS.m) | Feature ranking, swarm updates, and elite archive management |
| [SU_TOOL.m](SU_TOOL.m) | Symmetrical uncertainty computed from quantile-based feature bins |
| [jFitnessFunction.m](jFitnessFunction.m) | Cross-validated subset evaluation |
| [jknn.m](jknn.m) | Deterministic Euclidean nearest-neighbor classification |
| [verify_reproduction.m](verification/verify_reproduction.m) | Comparison with archived Case A results |
