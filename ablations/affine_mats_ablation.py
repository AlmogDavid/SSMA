import glob
import os

import numpy as np
import pandas as pd

from ssma import SSMA

"""
Training finished
Best train loss: 0.1634
Best val loss: 0.3147
Best test loss: 0.2940
Best train MAE: 0.1634
Best val MAE: 0.3147
Best test MAE: 0.2940
Summary results:
       index        lr  train_loss  val_loss  test_loss  best_train_loss  \
count    1.0  1.00e+00        1.00      1.00       1.00             1.00   
mean   421.0  1.95e-06        0.13      0.32       0.29             0.16   
std      NaN       NaN         NaN       NaN        NaN              NaN   
min    421.0  1.95e-06        0.13      0.32       0.29             0.16   
25%    421.0  1.95e-06        0.13      0.32       0.29             0.16   
50%    421.0  1.95e-06        0.13      0.32       0.29             0.16   
75%    421.0  1.95e-06        0.13      0.32       0.29             0.16   
max    421.0  1.95e-06        0.13      0.32       0.29             0.16   

       best_val_loss  best_test_loss  MAE_train  MAE_val  MAE_test  \
count           1.00            1.00       1.00     1.00      1.00   
mean            0.31            0.29       0.13     0.32      0.29   
std              NaN             NaN        NaN      NaN       NaN   
min             0.31            0.29       0.13     0.32      0.29   
25%             0.31            0.29       0.13     0.32      0.29   
50%             0.31            0.29       0.13     0.32      0.29   
75%             0.31            0.29       0.13     0.32      0.29   
max             0.31            0.29       0.13     0.32      0.29   

       best_MAE_train  best_MAE_val  best_MAE_test   step  
count            1.00          1.00           1.00    1.0  
mean             0.16          0.31           0.29  421.0  
std               NaN           NaN            NaN    NaN  
min              0.16          0.31           0.29  421.0  
25%              0.16          0.31           0.29  421.0  
50%              0.16          0.31           0.29  421.0  
75%              0.16          0.31           0.29  421.0  
max              0.16          0.31           0.29  421.0  

"""

if __name__ == "__main__":
    FOLDER = "/Users/almogdavid/Downloads/affine_mats/zinc_gcn_multiset_0"
    all_npy = glob.glob(os.path.join(FOLDER, "*.npy"))

    exp_name = os.path.basename(FOLDER)

    npy_by_layers = {}
    for npy_file in all_npy:
        file_base_name = os.path.basename(npy_file).split(".")[0]
        _, mat_type, layer_idx = file_base_name.split("_")

        data = np.load(npy_file)

        if layer_idx not in npy_by_layers:
            npy_by_layers[layer_idx] = [None, None]
        if mat_type == "mat":
            npy_by_layers[layer_idx][0] = data
        elif mat_type == "bias":
            npy_by_layers[layer_idx][1] = data
        else:
            raise RuntimeError(f"Invalid mat type: {mat_type}")
    ssma = SSMA(31, num_neighbors=3)
    orig_affine_w = ssma._affine_layer.weight.detach().numpy()
    orig_affine_b = ssma._affine_layer.bias.detach().numpy()
    orig_transform = np.concatenate([orig_affine_w.T, orig_affine_b.reshape(1, -1)], axis=0)

    target_excel = os.path.join(FOLDER, f"{exp_name}_affine_mats.xlsx")
    if os.path.exists(target_excel):
        os.remove(target_excel)
    print(f"Saving to: {target_excel}")

    all_orig_mats = []
    all_learned_mats = []

    for layer_idx, (layer_mat, layer_bias) in npy_by_layers.items():
        conc_arr = np.concatenate([layer_mat.T, layer_bias.reshape(1, -1)], axis=0)
        all_learned_mats.append(conc_arr)
        all_orig_mats.append(orig_transform)

        print(f"Layer {layer_idx} avg delta {np.mean(np.abs(conc_arr - orig_transform))}")

    all_orig_mats = np.stack(all_orig_mats)
    all_learned_mats = np.stack(all_learned_mats)
    all_mats_delta = np.abs(all_orig_mats - all_learned_mats)

    # Now compute stat
    avg_delta = np.mean(all_mats_delta)
    std_delta = np.std(all_mats_delta)

    avg_size = np.mean(np.abs(all_learned_mats))
    std_size = np.std(np.abs(all_learned_mats))

    percentage_change = (avg_delta / avg_size) * 100

    print(f"Average delta: {avg_delta}")
    print(f"STD delta: {std_delta}")
    print(f"Average size: {avg_size}")
    print(f"STD size: {std_size}")
    print(f"Percentage change: {percentage_change}")







