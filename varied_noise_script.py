"""Run varied noise experiments."""
from varied_noise import run_varied_noise_experiments

# Set datasets and noise types to True to run varied noise level experiments of said type on said datasets.
run_varied_noise_experiments(
    sklearn=True,
    wave=True,
    mnist=True,
    sst=True,
    normal_white_noise=True,
    uniform_white_noise=True,
    outliers=True,
    sparse_noise=True,
    output_folder="",
)
