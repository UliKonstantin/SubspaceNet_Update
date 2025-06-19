import matplotlib.pyplot as plt
from pathlib import Path
import logging

def plot_loss_vs_scenario(scenario_results, scenario, output_dir):
    """
    Plot ESPRIT and DNN loss vs. scenario values and save the plot.
    """
    logger = logging.getLogger("SubspaceNet.plotting")
    x_vals = list(scenario_results.keys())
    esprit_losses = []
    dnn_losses = []
    for v in x_vals:
        res = scenario_results[v]
        # If result is a float, treat as ESPRIT loss
        if isinstance(res, float) or isinstance(res, int):
            esprit_loss = res
            dnn_loss = None
        elif isinstance(res, dict):
            esprit_loss = None
            if 'evaluation_results' in res and 'classic_methods_test_losses' in res['evaluation_results'] and 'ESPRIT' in res['evaluation_results']['classic_methods_test_losses']:
                esprit_loss = res['evaluation_results']['classic_methods_test_losses']['ESPRIT']
            dnn_loss = res['evaluation_results'].get('dnn_test_loss')
        else:
            esprit_loss = None
            dnn_loss = None
        esprit_losses.append(esprit_loss)
        dnn_losses.append(dnn_loss)
        logger.debug(f"eta={v}: ESPRIT loss={esprit_loss}, DNN loss={dnn_loss}")
    if all(l is None for l in esprit_losses) and all(l is None for l in dnn_losses):
        logger.warning(f"All losses are None for scenario {scenario}. Plot will be empty.")
    plt.figure(figsize=(8, 6))
    if any(l is not None for l in esprit_losses):
        plt.plot(x_vals, esprit_losses, '-o', label='ESPRIT loss')
    if any(l is not None for l in dnn_losses):
        plt.plot(x_vals, dnn_losses, '-o', label='DNN loss')
    plt.xlabel(scenario)
    plt.ylabel('Loss')
    plt.title(f'Loss vs. {scenario}')
    plt.legend()
    plt.grid(True)
    plot_path = Path(output_dir) / f"loss_vs_{scenario}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path 