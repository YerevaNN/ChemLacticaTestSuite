from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

def get_scatter_title(config_name,test_name,model_checkpoint_path,rmse,mape,rmse_c,mape_c,correlation,correlation_c,n_invalid,n_total,n_unique=None,n_in_pubchem=None,sm=""):

    title = f'{config_name} generation of {test_name} with {model_checkpoint_path.split("/")[-2]}\n'\
            f'rmse {rmse:.3f} mape {mape:.3f} rmse_c {rmse_c:.3f} mape_c {mape_c:.3f}\n'\
            f'corr: {correlation:.3f} corr_c: {correlation_c:.3f} corr_s: {correlation*(1-(n_invalid/n_total)):.3f}\n'\
            f'N invalid: {n_invalid}, N total: {n_total}'

    for info in [n_unique,n_in_pubchem]: # If we include this information, include it in the title
        if info is not None:
            title += f", {info}"
    title+=sm

    return title


def get_scatter_plot_bounds(targets,generated_clean):
    max_, min_, max_g = np.max(targets), np.min(targets), np.max(generated_clean)
    return max_, min_, max_g


def clean_outputs(test_name, targets, property_range, calculated_properties):
    target_clean, generated_clean, nones = [], [], []
    corrected_calculated = np.array(calculated_properties)
    corrected_calculated[corrected_calculated == None] = property_range[test_name]['mean']
    for target, c_props in zip(targets, calculated_properties):
        # target *= self.generation_decoding_config["num_return_sequences"]
        for t, cp in zip(target, c_props):
            if  cp != None:
                target_clean.append(t)
                generated_clean.append(cp)
            else:
                nones.append(t)
    return target_clean, generated_clean, nones, corrected_calculated

def calculate_metrics(target,generated):
    mape = metrics.mean_absolute_percentage_error(target, generated)
    rmse = metrics.mean_squared_error(target, generated, squared=False)
    correlation, pvalue = spearmanr(target, generated)
    return rmse, mape, correlation

def paint_plot(title,test_name,stats,stats_width,target_clean,generated_clean,nones,min_,max_,diffs):
    fig, ax1 = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    fig.set_linewidth(4)
    ax2 = ax1.twinx()
    ax2.bar([interval.mid for interval in stats.index], stats, width=stats_width, alpha=0.3) 

    ax1.scatter(target_clean, generated_clean, c='b')
    ax1.vlines(nones, ymin=min_, ymax=max_, color='r', alpha=0.3)
    ax1.plot([min_, max_], [min_, max_], color='grey', linestyle='--', linewidth=2)
    ax1.plot(target_clean, np.convolve(np.pad(diffs, (2, 2), mode='edge'), np.ones(5)/5, mode='valid'), color='m', alpha=0.5)
    ax1.set_xlabel(f'target {test_name}')
    ax1.set_ylabel(f'generated {test_name}')
    ax1.grid(True)
    plt.title(title)
    plt.tight_layout()
    return fig




def make_plot(test_name, stats, property_range, targets, target_clean, generated_clean, nones, correlation, rmse, mape, correlation_c, rmse_c, mape_c):
    max_, min_, max_g = get_scatter_plot_bounds(targets, generated_clean)
    diffs = np.abs(np.array(target_clean) - np.array(generated_clean))
    title = get_scatter_title(
            config_name,
            test_name,
            model_checkpoint_path,
            rmse,
            mape,
            rmse_c,
            mape_c,
            correlation,
            correlation_c,
            n_invalid,
            n_total)

    stats_width = (property_range[1] - property_range[0]) / 100
    paint_plot(title,test_name,stats,stats_width,target_clean,generated_clean,nones,min_,max_,diffs)

def full_workflow(test_name,stats,targets,generated,calculated_properties):
    property_range = [0, 1]
    target_clean, generated_clean, nones, corrected_calculated = clean_outputs(test_name,targets, property_range, calculated_properties)
    rmse_c, mape_c, correlation_c = calculate_metrics(targets,corrected_calculated)
    rmse, mape, correlation = calculate_metrics(target_clean, generated_clean)
    make_plot(test_name, stats, property_range, targets, target_clean, generated_clean, nones, correlation, rmse, mape, correlation_c, rmse_c, mape_c)


