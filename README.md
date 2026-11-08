# Grid Audit

Code and recorded results for the article **"Rounding-Aware Graph Neural Diagnosis of Power-Flow Benchmark Errors with Mathematical Verification"** by Christopher John T. Raagas, Abid Yahya, Lory Bulay-og, Oliver Lexter July Jose, Eva Asiimwe and Irfan Anjum Badruddin.

The method ranks suspect circuit records in rounded power-flow benchmarks with an edge-gated graph neural network, then checks each inspected correction with a separate residual verifier. The learned ranking only decides the inspection order. Lower bounds and feasible operating points decide whether a candidate is compatible.

## Repository layout

| Path | Contents |
|---|---|
| `work/analysis.py` | Power-flow equations, analytic Jacobian, Newton solver, reference row-remainder bound and rounding helpers |
| `learning/study.py` | Network loading, candidate inventory, mutations, rounding-aware features, scenario generation, the graph models, baselines and training |
| `learning/repeat.py` | Repeat training runs with seeds 47 and 83 |
| `learning/audit.py` | Invariance and gradient checks, calibration, conditional fault-type confusion, source-repair verification and branch-group bootstrap |
| `learning/pipeline.py` | Coupled linear-programming bound and the queue-based source-record audit |
| `learning/physics-check.py` | Comparison of the solver and Jacobian with PYPOWER, and of the two remainder implementations |
| `learning/plots.py` | Regenerates the result figures |
| `recorded-results/` | JSON summaries from the completed experiment runs reported in the article |
| `figures/` | Recorded result plots and the circuit illustration |
| `run.py` | Command-line entry point |

## Setup

Use Python 3.12 in a virtual environment. The pinned numerical packages in `requirements.txt` record the original experiment environment. Matplotlib, Pillow and NetworkX are additional plotting dependencies; their original versions were not recorded.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python run.py --help
```

## Reproducing the results

Run all commands from the repository root. Full reproduction regenerates the scenarios and retrains the models; pretrained checkpoints and serialized scenario arrays are not bundled.

```bash
python run.py train      # scenarios, models and baselines -> learning/results.json
python run.py repeat     # repeat runs -> learning/repeats.json
python run.py audit      # checks, calibration and bootstrap -> learning/audit.json
python run.py pipeline   # source-record audit -> learning/pipeline.json
python run.py physics    # solver and remainder checks -> learning/physics-check.json
python run.py plots      # figures -> figures/
```

`python run.py all` runs the stages in this order. Training writes `learning/data.pkl`, the model checkpoints, `learning/evaluated.pkl` and numerical summaries. The repeat and audit stages must complete before the plots are regenerated. The physics check can run on its own after installing the dependencies. Training skips data generation if `learning/data.pkl` already exists; delete that file to regenerate the scenarios. Generated summaries are written under `learning/`, and the archived summaries stay in `recorded-results/` for comparison.

Random seeds are fixed: 4501 for scenario generation, 29 for the main training run and 47 and 83 for the repeat runs. Floating-point libraries and hardware may still change exact training results slightly.

## Experimental scope

The experiments use the PYPOWER 14-, 30-, 57- and 118-bus cases. The first three networks supply training, validation and test scenarios, split by corrupted circuit record; the whole 118-bus network is held out for transfer evaluation. Each scenario contains one controlled encoding mutation (missing circuit, reactance ×10 or ×0.1, resistance ×10, charging removed or doubled) and rounded voltage and angle observations. Accepted scenario counts are 2,100 training, 480 validation, 660 withheld-record tests and 350 transfer tests.

The primary edge-aware model reaches 84.5% top-1 localization across the 660 withheld-record tests and 72.9% on the unseen network, against 69.4% and 62.3% for residual ranking. A random forest is stronger on aggregate top-1 accuracy, so the results should be read together with the baselines, fault-family breakdowns and repeat runs rather than as universal superiority.

Neural rankings prioritize source inspection; they do not certify a candidate. The source-repair replay assumes access to trusted simulator source records. It is not a field demonstration or an autonomous correction guarantee.

## Checks on this distribution

All Python files compile. The physics-check stage was re-run on this repository with PYPOWER 5.1.21 and reproduced every value in `recorded-results/physics-check.json`. The complete training pipeline was executed during the original study and was not retrained while preparing this repository.

## Data provenance

Network cases are loaded through PYPOWER 5.1.21 (https://github.com/rwl/PYPOWER), which is distributed under its own licence. No network data are redistributed here.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`). Third-party packages and the PYPOWER cases keep their own terms.

## Citation

If you use this code, please cite the article above and this repository. Citation metadata for the repository is in `CITATION.cff`; the article reference will be added here after publication.

## Contact

Corresponding author: Eva Asiimwe, Kampala International University (eva.assiimwe@kiu.ac.ug).
