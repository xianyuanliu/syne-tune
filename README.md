# Optimizing Hyperparameters with Conformal Quantile Regression

This page describes how to reproduce the results from the following paper:

```
Optimizing Hyperparameters with Conformal Quantile Regression.
Salinas, Golebiowski, Klein, Seeger, Archambeau. ICML 2023.
```

To run all experiments, you can run the following:

```bash
git clone https://github.com/geoalgo/syne-tune.git -b icml_conformal
cd syne-tune
export PYTHONPATH="${PYTHONPATH}:${PWD}"
pip install -e ".[extra]"
pip install -r benchmarking/nursery/benchmark_conformal/requirements.txt
python benchmarking/nursery/benchmark_conformal/benchmark_main.py --experiment_tag "my-new-experiment" --num_seeds 30
```

Which will run all combinations of methods/benchmark/seeds on your local computer for 30 seeds.

Once all evaluations are done, you can plot results by running:

```python benchmarking/nursery/benchmark_conformal/results_analysis/show_results.py --experiment_tag "my-new-experiment"``` 

you will obtain all plots and table of the paper including performance over time of all methods on all benchmarks
and also a table showing the average ranks/normalized scores of methods.

You can also run only one scheduler by doing `python benchmarking/nursery/benchmark_conformal/benchmark_main.py --method RS  --experiment_tag "my-new-experiment"`, see
run `python benchmarking/nursery/benchmark_conformal/benchmark_main.py --help` to see all options supported.

**Evaluation code structure.**
To evaluate other methods/benchmarks, you can edit the following files:
* `baselines.py`: dictionary of HPO methods to be evaluated 
* `benchmark_definitions.py`: dictionary of simulated benchmark to evaluate
* `benchmark_main.py`: script to launch evaluations, run all combinations by default
* `launch_remote.py`: script to launch evaluations on a remote instance
* `show_results.py`: script to plot results obtained 
* `requirements.txt`: dependencies to be installed when running locally or on a remote machine.

**Running on AWS SageMaker.** 
To launch the evaluation remotely on SageMaker, you can also run 
```python benchmarking/nursery/benchmark_conformal/launch_remote.py --experiment_tag "my-new-experiment"``` which 
evaluate combinations in parallel using remote machines, this will requires that you have set up an AWS account, 
you can check [here](https://syne-tune.readthedocs.io/en/latest/faq.html#how-can-i-run-on-aws-and-sagemaker) 
for instance for instructions.

If so, you will need to sync remote results if you want to analyse them locally which can be done with:

```bash
aws s3 sync s3://{YOUR_SAGEMAKER_BUCKET}/syne-tune/{experiment_tag}/ ~/syne-tune/
```

If you dont know your sagemaker-bucket, you can get it by running:
```
aws s3 ls | grep sagemaker
```
which should shown something like `sagemaker-us-west-2-123456789101` where `123456789101` is your account-id.

