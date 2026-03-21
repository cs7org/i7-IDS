from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
import numpy as np
from loguru import logger
from pathlib import Path

normal_data_root = Path('/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_blocking_mobnet/mobilenet_v3_large_nosampling')
normalized_data_root = Path('/home/hpc/iwi7/iwi7101h/i7-IDS/results/adversarial_blocking_mobnet/mobilenet_v3_large_normalized_nosampling')

def calculate_thresholded_accuracy(data_root, use_max_diff=False):
    all_mae_files = list(data_root.rglob('*.npy'))
    results = []

    all_mae_values = dict()
    all_mae_eps_values = dict()  # Store MAE values for each (method, epsilon) combination

    for mae_file in all_mae_files:
        # wrong file
        if 'mae_values'==mae_file.stem:
            continue
        # extract method and epsilon from filename
        epsilon = mae_file.parent.name.split('eps_')[-1]
        method = mae_file.parent.name
        method = 'MIM' if 'momentum' in method else 'FGSM' if 'fast' in method else 'BIM' if 'basic' in method else 'Unknown'
        dtype = 'Normalized' if 'normalized' in str(mae_file).lower() else 'Normal'
        defender = 'UNet' if 'custom' in str(mae_file).lower() else 'RDU-Net' 

        mae_data= np.load(mae_file)

        # contains: [N, 4], clean mae, adv recon mae, clean recon max diff, adv recon max diff
        idx = len(mae_data)
        if use_max_diff:
            clean_recon_mae_values, adv_recon_mae_values = mae_data[:, 2], mae_data[:, 3]
        else:
            clean_recon_mae_values, adv_recon_mae_values = mae_data[:, 0], mae_data[:, 1]
        if all_mae_values.get(defender) is None:
            all_mae_values[defender] = clean_recon_mae_values.tolist() + adv_recon_mae_values.tolist()
        else:
            all_mae_values[defender].extend(adv_recon_mae_values.tolist())
        
        all_mae_eps_values[(defender, method, epsilon)] = (clean_recon_mae_values, adv_recon_mae_values)
        

    # now calculate overall threshold across all methods and epsilons
    def_thresh = {}
    for defender, mae_values in all_mae_values.items():
        all_clean_mae_values = np.array(mae_values[:idx])
        all_adv_mae_values = np.array(mae_values[idx:])

        y_true = np.concatenate([np.zeros_like(all_clean_mae_values), np.ones_like(all_adv_mae_values)])
        y_scores = np.concatenate([all_clean_mae_values, all_adv_mae_values])
        

        auc = roc_auc_score(y_true, y_scores)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]
        y_pred = (y_scores > best_threshold).astype(int)

        best_auc = roc_auc_score(y_true, y_scores)
        best_f1 = f1_score(y_true, y_pred)
        best_accuracy = accuracy_score(y_true, y_pred)
        def_thresh[defender] = best_threshold

        logger.info(f"Overall Best threshold for defender {defender}: {def_thresh[defender]}, Overall Best AUC: {best_auc}")
        logger.info(f"Defender {defender} | Overall TPR: {tpr[best_idx]}, FPR: {fpr[best_idx]}, F1: {best_f1}, Accuracy: {best_accuracy}")

        # Add overall results to the results list
        results.append({
            'method': 'Overall',
            'epsilon': 'All',
            'dtype': dtype,
            'defender': defender,
            'best_threshold': def_thresh[defender],
            'best_auc': best_auc,
            'best_f1': best_f1,
            'best_accuracy': best_accuracy
        })
    
    for (defender, method, epsilon), (clean_mae_values, adv_mae_values) in all_mae_eps_values.items():
        y_true = np.concatenate([np.zeros_like(clean_mae_values), np.ones_like(adv_mae_values)])
        y_scores = np.concatenate([clean_mae_values, adv_mae_values])
        
        best_threshold =def_thresh[defender]  # Use the overall best threshold for this defender
        y_pred = (y_scores > best_threshold).astype(int)
        best_auc = roc_auc_score(y_true, y_pred)
        best_f1 = f1_score(y_true, y_pred)
        best_accuracy = accuracy_score(y_true, y_pred)

        print(f" Defender: {defender}, Method: {method}, Epsilon: {epsilon}, Best threshold: {best_threshold}, Best AUC: {best_auc}, Best F1: {best_f1}, Best Accuracy: {best_accuracy}")
        # Add method-specific results to the results list
        results.append({
            'method': method,
            'epsilon': epsilon,
            'dtype': dtype,
            'defender': defender,
            'best_threshold': best_threshold,
            'best_auc': best_auc,
            'best_f1': best_f1,
            'best_accuracy': best_accuracy
        })

    # save results to csv
    results_df = pd.DataFrame(results)
    if use_max_diff:
        results_df.to_csv(data_root / 'adversarial_detection_results_max_diff.csv', index=False)
        logger.info(f"Results saved to {data_root / 'adversarial_detection_results_max_diff.csv'}")
    else:
        results_df.to_csv(data_root / 'adversarial_detection_results.csv', index=False)
        logger.info(f"Results saved to {data_root / 'adversarial_detection_results.csv'}")
    
    return results_df
    

def plot_curves(normal_df, norm_df, plot_key='best_accuracy'):    
    # Get unique epsilons from both dataframes, ensure 'All' is first
    epsilons_normal = normal_df['epsilon'].unique().tolist()
    epsilons_norm = norm_df['epsilon'].unique().tolist()
    epsilons = sorted(set(epsilons_normal + epsilons_norm), key=lambda x: (x != 'All', float(x) if x != 'All' else -1))

    defenders = ['RDU-Net', 'UNet']
    methods = ['FGSM', 'BIM', 'MIM']
    bar_width = 0.13

    # 6 UNIQUE colors for each defender-method combination
    bar_colors = {
        'RDU-Net-FGSM': '#1f77b4',    # Blue
        'RDU-Net-BIM': '#ff7f0e',     # Orange  
        'RDU-Net-MIM': '#2ca02c',     # Green
        'UNet-FGSM': '#d62728',       # Red
        'UNet-BIM': '#9467bd',        # Purple
        'UNet-MIM': '#8c564b'         # Brown
    }

    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 
                        'axes.titlesize': 17, 'xtick.labelsize': 12, 
                        'ytick.labelsize': 12, 'legend.fontsize': 10})

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    for row, (df, dtype) in enumerate([(normal_df, 'Normal'), (norm_df, 'Normalised')]):
        ax = axes[row]
        x_labels = epsilons
        x_pos = np.arange(len(x_labels))
        
        # Plot 6 bars per epsilon (2 defenders × 3 methods)
        for def_idx, defender in enumerate(defenders):
            for meth_idx, method in enumerate(methods):
                bar_idx = def_idx * 3 + meth_idx
                key = f"{defender}-{method}"
                accs = []
                for eps in x_labels:
                    if eps == 'All':
                        sel = df[(df['defender'] == defender) & (df['method'] == 'Overall') & (df['epsilon'] == 'All')]
                        acc = sel[plot_key].values[0] if not sel.empty else np.nan
                    else:
                        sel = df[(df['defender'] == defender) & (df['method'] == method) & (df['epsilon'] == eps)]
                        acc = sel[plot_key].values[0] if not sel.empty else np.nan
                    accs.append(acc)
                offset = (bar_idx - 2.5) * bar_width
                ax.bar(x_pos + offset, accs, width=bar_width, 
                    color=bar_colors[key], alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        y_label = 'Accuracy' if plot_key == 'best_accuracy' else 'F1 Score' if plot_key == 'best_f1' else 'AUC'
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(f'{dtype}', fontsize=16, pad=10)
        ax.set_ylim(0, 1)
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Remove left/right padding
        ax.margins(x=0)
        ax.set_xlim(-0.5, len(x_labels)-0.5)

    axes[1].set_xlabel('Epsilon', fontsize=14)

    # Legends - MOVED DOWN
    handles_rdu = [plt.Rectangle((0,0),1,1, color=bar_colors['RDU-Net-FGSM']),
                plt.Rectangle((0,0),1,1, color=bar_colors['RDU-Net-BIM']),
                plt.Rectangle((0,0),1,1, color=bar_colors['RDU-Net-MIM'])]
    handles_unet = [plt.Rectangle((0,0),1,1, color=bar_colors['UNet-FGSM']),
                    plt.Rectangle((0,0),1,1, color=bar_colors['UNet-BIM']),
                    plt.Rectangle((0,0),1,1, color=bar_colors['UNet-MIM'])]

    rdu_labels = ['FGSM', 'BIM', 'MIM']
    unet_labels = ['FGSM', 'BIM', 'MIM']

    # **LEGENDS MOVED LOWER** - y=0.92 instead of 0.98
    fig.legend(handles_rdu, rdu_labels, title='RDU-Net', loc='upper center', 
            bbox_to_anchor=(0.27, 0.983), ncol=3, fontsize=10, title_fontsize=11,
            frameon=True, fancybox=False, edgecolor='black', columnspacing=1)
    fig.legend(handles_unet, unet_labels, title='UNet', loc='upper center',
            bbox_to_anchor=(0.818, 0.983), ncol=3, fontsize=10, title_fontsize=11,
            frameon=True, fancybox=False, edgecolor='black', columnspacing=1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjusted for lower legends
    return fig




if __name__ == "__main__":
    logger.info("Calculating MAE thresholds for normal data...")
    normal_res_df = calculate_thresholded_accuracy(normal_data_root)
    logger.info("Calculating MAE thresholds for normalized data...")
    normalised_res_df = calculate_thresholded_accuracy(normalized_data_root)

    normal_df = pd.read_csv(normal_data_root/'adversarial_detection_results.csv')
    norm_df = pd.read_csv(normalized_data_root/'adversarial_detection_results.csv')

    logger.info("Loading results for plotting...")
    fig = plot_curves(normal_df, norm_df, plot_key='best_accuracy')
    plt.savefig('adversarial_detection_accuracy.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # now for diff
    normal_res_df = calculate_thresholded_accuracy(normal_data_root, use_max_diff=True)
    normalised_res_df = calculate_thresholded_accuracy(normalized_data_root, use_max_diff=True)

    normal_df = pd.read_csv(normal_data_root/'adversarial_detection_results_max_diff.csv')
    norm_df = pd.read_csv(normalized_data_root/'adversarial_detection_results_max_diff.csv')

    logger.info("Loading max diff results for plotting...")
    fig = plot_curves(normal_df, norm_df)
    plt.savefig('adversarial_detection_max_diff.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# Normalised THRESH: UNet 0.882, RDU-Net 0.089
# Normal THRESH: UNet 0.812, RDU-Net 0.28
# Normal: RDUNet: TPR: 0.83, FPR: 0.0 | UNet: TPR: 0.714, FPR: 0.0298
# Normalised: RDUNet TPR: 0.83, FPR: 0.0 | TPR: 0.756, FPR: 0.0572