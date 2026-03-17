# DARE: Data-Aware Monte Carlo Precision Error Estimation

This repository contains the implementation of **DARE**, a data-aware Monte Carlo framework for estimating, diagnosing, and explaining precision errors in mixed-precision tensor operators.

DARE is designed to answer three practical questions:

1. **How large can the precision error become?**
2. **Did the observed runtime error exceed the predicted safe bound?**
3. **Which stage of the operator is responsible for most of the error?**

The codebase includes operator-specific pipelines for Conv2D, MatMul, GEMM, Linear, LayerNorm, BatchNorm, Softmax, ReLU, pooling, and attention.

---

## Environment Setup

We recommend:

- Python `>= 3.9`
- PyTorch `>= 2.0`
- CUDA-enabled PyTorch if you want to use GPU-backed Monte Carlo sampling

Example setup:

```bash
conda create -n dare python=3.9 -y
conda activate dare

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm
```

Optional packages for evaluation and plotting:

```bash
pip install seaborn pandas
```

---

## Repository Structure

```text
precision_estimation/
|-- core/
|   |-- analyzer/      # Interprets oracle outputs and attributes error sources
|   |-- config/        # Precision strategy definitions and quantization helpers
|   |-- detector/      # Compares predicted bounds with actual runtime errors
|   |-- generator/     # Creates operator inputs, weights, and metadata
|   `-- oracle/        # Monte Carlo error-bound prediction for each operator
|-- examples/          # End-to-end operator pipelines
|-- evaluation/        # Result aggregation, reporting, and visualization
|-- data/              # Additional assets or intermediate data
|-- figures/           # Generated figures
|-- tables/            # Generated tables
`-- *.log              # Experiment logs
```

---

## Pipeline Workflow

Each script in `examples/*_precision_pipeline.py` follows the same runtime pattern. The Conv2D pipeline is a representative example, but the same structure is reused across other operators.

### End-to-End Flow

```text
Precision Strategy
    -> Input Generator
    -> Monte Carlo Oracle
    -> Error Detector
    -> Error Analyzer
    -> JSON Results / Logs / Evaluation Reports
```

### What Each Component Does

1. **Precision strategy (`core/config/`)**

   Defines how the operator is executed under mixed precision:

   - input storage dtype
   - weight storage dtype
   - compute dtype
   - output dtype
   - optional fake-quantization behavior

   This is the global execution policy used by the oracle, detector, and analyzer.

2. **Input generator (`core/generator/`)**

   Produces operator-specific tensors and metadata, for example:

   - input activations
   - weights or affine parameters
   - operator hyperparameters such as stride, padding, or normalization shape
   - distribution settings such as `normal`, `boundary`, or `adversarial_sum`

   The generator is responsible for creating the data instances on which precision behavior is tested.

3. **Monte Carlo oracle (`core/oracle/`)**

   Predicts an error bound before comparing against the real execution result.

   In general, the oracle:

   - simulates storage, compute, accumulation, and output-rounding noise
   - runs repeated Monte Carlo trials
   - computes a high-precision reference result
   - collects sampled error magnitudes
   - estimates a high-quantile bound with a safety factor
   - produces component-wise error estimates such as input-storage error, weight-storage error, accumulation error, and output demotion error

   This is the main predictive module in DARE.

4. **Error detector (`core/detector/`)**

   Executes the operator once through the selected mixed-precision path and once through a high-precision reference path, then compares the two.

   The detector answers:

   - what the actual maximum error was
   - what bound the oracle predicted
   - whether the observed error exceeded the bound

   In other words, the detector is the validation layer between prediction and actual execution.

5. **Error analyzer (`core/analyzer/`)**

   Explains *why* the error happened.

   Depending on the operator, the analyzer can:

   - summarize component-wise error ratios
   - identify the dominant error source
   - inspect top error locations or critical paths
   - provide operator-specific suggestions, such as increasing accumulation precision or delaying output demotion

   This stage turns the bound estimate into an interpretable diagnosis.

6. **Evaluation and artifacts (`evaluation/`)**

   Pipeline scripts save per-run JSON files, which can later be aggregated by the evaluation utilities.

   The evaluation layer:

   - loads many pipeline outputs
   - computes metrics such as coverage rate and bound tightness
   - summarizes cross-operator and cross-strategy behavior
   - generates plots and report files

### Runtime View of a Typical Pipeline

For a script such as `examples/conv2d_precision_pipeline.py`, the workflow is:

1. Select a precision strategy, for example `BF16_compute`.
2. Generate a batch of operator inputs and metadata.
3. Build the corresponding Monte Carlo oracle.
4. Run the detector to obtain:
   - predicted bound
   - actual observed error
   - exceed/not-exceed decision
   - raw oracle result
5. Pass the oracle result and original tensors to the analyzer.
6. Save structured JSON output for later evaluation.

This means an operator pipeline in DARE is not just a single inference run. It is a full loop that combines **data generation, error prediction, empirical validation, attribution, and result export**.

---

## Quick Start

Run any operator pipeline from the `examples/` directory. For example:

```bash
python examples/conv2d_precision_pipeline.py
```

Other common entry points:

```bash
python examples/layernorm_precision_pipeline.py
python examples/matmul_precision_pipeline.py
python examples/softmax_precision_pipeline.py
python examples/attention_precision_pipeline.py
```

---

## Expected Outputs

A typical pipeline run produces some combination of the following:

- console logs with predicted bounds and actual errors
- JSON analysis files under `evaluation/experiment_*`
- experiment logs such as `*_precision.log`
- plots and summary reports generated by evaluation scripts

The saved JSON files usually contain:

- detector outputs such as `actual_err`, `predicted_bound`, and elapsed time
- analyzer outputs such as component estimates, component ratios, primary source, and top-path summaries

---

## Running the Evaluation Suite

After collecting JSON outputs from one or more operator pipelines, run:

```bash
python evaluation/precision_evaluation_analyzer.py
```

The evaluation code aggregates experimental results and reports metrics such as:

- coverage rate of predicted bounds
- average and median bound tightness
- runtime efficiency
- consistency across operators and precision strategies
- attribution quality trends

It can also generate figures and summary report files, for example:

- `dare_evaluation_report.json`
- `dare_evaluation_report_all_strategies.json`
- `plot_error_prediction_accuracy.*`
- `plot_bound_tightness_distribution.*`
- `plot_operator_efficiency.*`
- `plot_primary_error_sources.*`

---

## Notes

- Most pipelines are operator-specific, but their control flow is intentionally consistent.
- Monte Carlo sampling counts, quantiles, and safety factors can be adjusted inside each example script or oracle constructor.
- GPU execution is optional; if CUDA is available, several oracles can distribute sampling across multiple devices.
