import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os
from datetime import datetime

########################## Data Loading ################################

def get_num_examples_in_block(block_len, seq_len):
    return max(block_len - seq_len + 1, 0)

def get_num_examples_in_data(block_len, seq_len, total_size):
    num_full_blocks = total_size // block_len
    block_remainder = total_size % block_len
    num_examples = num_full_blocks * get_num_examples_in_block(block_len, seq_len) \
                 + get_num_examples_in_block(block_remainder, seq_len)
    return num_examples

def load_and_preprocess_sequence_data(data_file, feature_cols, days_in_train_block, days_in_val_block, days_in_test_block, 
                                      seq_len, device):
    # read in the processed data. there should be zero nans or missing data.
    CAISO_Data = pd.read_csv(data_file, index_col=0)
    CAISO_Data.index = pd.to_datetime(CAISO_Data.index)

    # Adding some temporal features
    CAISO_Data.loc[:,"Day_of_Year"] = [instant.timetuple().tm_yday for instant in CAISO_Data.index]
    CAISO_Data.loc[:,"Hour"] = CAISO_Data.index.hour

    num_samples = len(CAISO_Data)

    # create train, validation and test data
    data_sets = {"train": {"days_in_block": days_in_train_block}, 
                 "val": {"days_in_block": days_in_val_block}, 
                 "test": {"days_in_block": days_in_test_block}}
    for data in data_sets.values():
        data["block_len"] = data["days_in_block"] * 24 + seq_len - 1
    set_assignments = []
    remaining_samples_to_assign = num_samples
    cur_set_idx = 0
    set_names = list(data_sets.keys())
    while remaining_samples_to_assign:
        cur_set = set_names[cur_set_idx]
        samples_to_assign = min(remaining_samples_to_assign, data_sets[cur_set]["block_len"])
        set_assignments.extend([cur_set for i in range(samples_to_assign)])

        cur_set_idx = (cur_set_idx + 1) % len(set_names)
        remaining_samples_to_assign -= samples_to_assign

    set_assignments = np.array(set_assignments)
    for set_name, data in data_sets.items():
        data["mask"] = set_assignments == set_name

    points_in_last_assignment = 0
    while set_assignments[-(points_in_last_assignment + 1)] == set_assignments[-1]:
        points_in_last_assignment += 1
    last_assignment_set = set_assignments[-1]
#     print(f"The last block of points assigned is in the {last_assignment_set} set and has {points_in_last_assignment/24} day(s) of points.")

#     for set_name, data in data_sets.items():
#         print(f"number of points in {set_name} set: {sum(data['mask'])}")

    # specify x, bottleneck features, and y data
    bottleneck_feature_cols = ["delta_Load", "delta_VRE"]
    y_col = 'delta_Total_CO2_Emissions'

    for set_name, data in data_sets.items():
        CAISO_subset = CAISO_Data.loc[data["mask"]]
        data["X"] = CAISO_subset[feature_cols].values.astype(np.float32)
        data["y"] = CAISO_subset[y_col].values.astype(np.float32)
        data["bottleneck_X"] = CAISO_subset[bottleneck_feature_cols].values.astype(np.float32)
    # add a "full" set containing the whole unsplit data from train/val/test in-order.
    data_sets["full"] = {"X": CAISO_Data[feature_cols].values.astype(np.float32),
                         "y": CAISO_Data[y_col].values.astype(np.float32),
                         "bottleneck_X": CAISO_Data[bottleneck_feature_cols].values.astype(np.float32),
                         "block_len": len(CAISO_Data),
                        }

    # standardize data based on mean and variance of train data
    scaler = preprocessing.StandardScaler()
    scaler.fit(data_sets["train"]["X"])
    for data in data_sets.values():
        data["X"] = scaler.transform(data["X"])

    # seperate data to contiguous blocks
    def reshape_to_blocks(arr, block_size):
        block_arr = []
        start_pos = 0
        while start_pos < len(arr):
            end = min(start_pos + block_size, len(arr))
            block_arr.append(arr[start_pos: end])
            start_pos = end
        return block_arr

    def split_blocks_to_seqs(blocks, seq_len=seq_len):
        """
        blocks: num_blocks x block_len [x features]

        ret:
          seqs: num_seqs x seq_len [x features]
        """
        seqs = []
        for block in blocks:
            block_len = len(block)
            assert block_len >= seq_len
            seqs.extend([block[start: start+seq_len] for start in range(block_len - seq_len + 1)])
        return np.array(seqs)

    # split data to contiguous blocks, and expand to example sequences via sliding window over blocks
    for data in data_sets.values():
        for key in ["X", "y", "bottleneck_X"]:
            data[key] = reshape_to_blocks(data[key], data["block_len"])
            data[key] = split_blocks_to_seqs(data[key], seq_len)
            data[key] = torch.tensor(data[key]).to(device)

    return data_sets


############## Helper functions for evaluating model predictions / plotting ####################

def get_y_pred(pred_coeff, bottleneck_X):

    MEF_preds = pred_coeff[:,:,0]
    MDF_preds = pred_coeff[:,:,1]
    delta_load = bottleneck_X[:,:,0]
    delta_vre = bottleneck_X[:,:,1]
    y_pred_demand = torch.mul(delta_load, MEF_preds)
    y_pred_vre = torch.mul(delta_vre, MDF_preds)
    y_pred = y_pred_vre + y_pred_demand
    if pred_coeff.shape[-1] == 3:
        # we are using a linear model with an intercept term
        y_pred += pred_coeff[:,:,2]
    
    return y_pred

def plot_losses(train_losses, val_losses, plt_save_dir=None, verbose=True):
    plt.ioff()
    # plot loss vs epochs
    fig, axs = plt.subplots(1,2)
    axs[0].plot(train_losses[50:])
    axs[0].set_title("Train Set")
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')

    axs[1].plot(val_losses[50:])
    axs[1].set_title("Val Set")
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    
    if verbose:
        plt.tight_layout()
        plt.show()
    
    if plt_save_dir:
        fig.savefig(os.path.join(plt_save_dir, "train_val_losses.png"))
        
    plt.close(fig)
        
def get_r_squared(pred_coeff, bottleneck_X, y):
    y_pred = get_y_pred(pred_coeff, bottleneck_X)
    # only one element in sequence is being predicted, return single R2
    if y.shape[-1] == 1:
        return r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    # more than one elt in seq being predicted, return an R2 for each position that we could evaluate at
    else:
        return [r2_score(y_i, y_pred_i) for y_i, y_pred_i in \
                zip(y.permute(1,0).cpu().detach().numpy(),
                    y_pred.permute(1,0).cpu().detach().numpy())]

def get_mean_abs_err(pred_coeff, bottleneck_X, y):
    y_pred = get_y_pred(pred_coeff, bottleneck_X)
    # only one element in sequence is being predicted, return single MAE
    if y.shape[-1] == 1:
        return mean_absolute_error(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    # more than one elt in seq being predicted, return an MAE for each position that we could evaluate at
    else:
        return [mean_absolute_error(y_i, y_pred_i) for y_i, y_pred_i in \
                zip(y.permute(1,0).cpu().detach().numpy(),
                    y_pred.permute(1,0).cpu().detach().numpy())]

def get_count_invalid_preds(pred_coeff):
    count_neg_MEFs = torch.sum(pred_coeff[:,:,0] <= 0).item()  # MEF must be greater than 0
    count_pos_MDFs = torch.sum(pred_coeff[:,:,1] > 0).item()  # MDF must be less than or equal to 0
    return count_neg_MEFs, count_pos_MDFs


##################################### Loss Functions #####################################

def mse_loss_l2_coeff_reg(pred_coeff, bottleneck_X, y, MEF_reg_weight, MDF_reg_weight):
    """
    pred_coeff: batch x seq_len x output_dim
    bottleneck_X: batch x seq_len x bottleneck_dim
    y: batch x seq_len
    """

    y_pred = get_y_pred(pred_coeff, bottleneck_X)

    MEF_preds = pred_coeff[:,:,0]
    MDF_preds = pred_coeff[:,:,1]

    # Compute MEF regularization term (sum(MEF^2 + intercept if MEF < 0 for MEF in examples))
    invalid_MEFs = torch.flatten(nn.functional.relu(-MEF_preds))  # keep negative MEFs and zero others
    count_invalid_MEFs = torch.count_nonzero(invalid_MEFs)
    MEF_reg_intercept = 4420  # based on an average value of -65 seen amongst invalids when trained without regularization
    MEF_reg = torch.dot(invalid_MEFs, invalid_MEFs) + (count_invalid_MEFs * MEF_reg_intercept)

    # Compute MDF regularization term (sum(MDF^2 + intercept if MDF > 0 for MDF in examples))
    invalid_MDFs = torch.flatten(nn.functional.relu(MDF_preds))  # keep negative MEFs and zero others
    count_invalid_MDFs = torch.count_nonzero(invalid_MDFs)
    MDF_reg_intercept = 538  # based on an average value of +23 seen amongst invalids when trained without regularization
    MDF_reg = torch.dot(invalid_MDFs, invalid_MDFs) + (count_invalid_MDFs * MDF_reg_intercept)

    loss = nn.MSELoss()(y_pred, y) + (MEF_reg_weight * MEF_reg) + (MDF_reg_weight * MDF_reg)
    return loss

def mse_loss_l1_coeff_reg(pred_coeff, bottleneck_X, y, MEF_reg_weight, MDF_reg_weight):
    """
    pred_coeff: batch x seq_len x output_dim
    bottleneck_X: batch x seq_len x bottleneck_dim
    y: batch x seq_len
    """
    y_pred = get_y_pred(pred_coeff, bottleneck_X)

    MEF_preds = pred_coeff[:,:,0]
    MDF_preds = pred_coeff[:,:,1]

    # Compute MEF regularization term (sum(MEF + intercept if MEF < 0 for MEF in examples))
    invalid_MEFs = nn.functional.relu(-MEF_preds)  # keep negative MEFs and zero others
    count_invalid_MEFs = torch.count_nonzero(invalid_MEFs)
    MEF_reg_intercept = 66.5  # The average value seen amongst invalids when trained without regularization
    MEF_reg = torch.sum(invalid_MEFs) + (count_invalid_MEFs * MEF_reg_intercept)

    # Compute MDF regularization term (sum(MDF + intercept if MDF > 0 for MDF in examples))
    invalid_MDFs = nn.functional.relu(MDF_preds)  # keep negative MEFs and zero others
    count_invalid_MDFs = torch.count_nonzero(invalid_MDFs)
    MDF_reg_intercept = 23.2  # The average value seen amongst invalids when trained without regularization
    MDF_reg = torch.sum(invalid_MDFs) + (count_invalid_MDFs * MDF_reg_intercept)

    loss = nn.MSELoss()(y_pred, y) + (MEF_reg_weight * MEF_reg) + (MDF_reg_weight * MDF_reg)
    return loss

######################################## Model Training ########################################

def train_model_with_params_batched(data_sets, data_settings,  # data
                            model, model_settings,  # model settings
                            learning_rate, weight_decay,  # optimizer settings
                            loss_function, MEF_reg_weight, MDF_reg_weight,  # loss function settings
                            model_dir, batch_size=None, epochs=10000, # train process settings
                            print_freq=1000, min_save_r2=.87, max_save_mae=150000, verbose=True):  # train process settings

    train_set, val_set = data_sets["train"], data_sets["val"]      
    train_X, train_bottleneck_X, train_y = train_set["X"], train_set["bottleneck_X"], train_set["y"]
    val_X, val_bottleneck_X, val_y = val_set["X"], val_set["bottleneck_X"], val_set["y"]
    
    if not batch_size:
        batch_size = len(train_X)
        
    # some of our logic in here depends on if we are predicting just the final point in the sequence, or all points in sequence
    final_point_only = model_settings["final_point_only"]

    # if predicting final point in sequence only, keep only those corresponding points
    # in the bottleneck_X and y arrays
    if final_point_only:
        train_bottleneck_X = torch.unsqueeze(train_bottleneck_X[:,-1,:], 1)
        val_bottleneck_X = torch.unsqueeze(val_bottleneck_X[:,-1,:], 1)
        train_y = torch.unsqueeze(train_y[:,-1], 1)
        val_y = torch.unsqueeze(val_y[:,-1], 1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set up folder where model and model info will be saved
    model_dir = os.path.join(model_dir, str(datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')))
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # log experiment settings
    settings_str = "Model Settings:"
    for name, val in model_settings.items():
        settings_str += f"\n\t{name}={val}"
    settings_str += f"\nModel Architecture:\n{model}"
    settings_str += "\nData Settings:"
    for name, val in data_settings.items():
        settings_str += f"\n\t{name}={val}"
    settings_str += "\nOptimizer Settings:"
    settings_str += f"\n\t{learning_rate=}\n\t{weight_decay=}"
    settings_str += "\nLoss Function Settings:"
    settings_str += f"\n\t{loss_function=}\n\t{MEF_reg_weight=}\n\t{MDF_reg_weight=}"
    settings_str += "\nTrain Process Settings:"
    settings_str += f"\n\t{batch_size=}\n\t{epochs=}\n\t{min_save_r2=}\n\t{max_save_mae=}"
    if verbose:
        print(settings_str)
    with open(os.path.join(model_dir, "experiment_settings.txt"), 'w+') as f:
        f.write(settings_str)

    # set up vars for keeping track of stats of best-seen models
    best_r2 = -np.inf
    best_r2_epoch = None
    best_mae = np.inf
    best_mae_epoch = None
    best_r2_model = {"r2": -np.inf}
    best_mae_model = {"mae": np.inf}
    if not final_point_only:
        best_r2_any_point = -np.inf
        best_r2_any_point_list = None
        best_mae_any_point = np.inf
        best_mae_any_point_list = None
    
    # train/eval loop
    train_losses = []
    val_losses = []
    batches_per_epoch = int(np.ceil(len(train_X) / batch_size))
    for epoch in tqdm(range(epochs)):
        if batches_per_epoch > 1:
            shuffle_perm = torch.randperm(len(train_X))
            train_X = train_X[shuffle_perm]
            train_bottleneck_X = train_bottleneck_X[shuffle_perm]
            train_y = train_y[shuffle_perm]

        # tell model we are training
        model.train()
        # make updates based on each batch in training data
        for batch in range(batches_per_epoch):
            batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size
            batch_train_X = train_X[batch_start: batch_end]
            batch_train_bottleneck_X = train_bottleneck_X[batch_start: batch_end]
            batch_train_y = train_y[batch_start: batch_end]
        
            batch_train_pred_coeff = model(batch_train_X.float())
            batch_train_loss = loss_function(batch_train_pred_coeff, batch_train_bottleneck_X, batch_train_y, MEF_reg_weight, MDF_reg_weight)
            model.zero_grad()
            batch_train_loss.backward()
            optimizer.step()
        del batch_train_pred_coeff
        del batch_train_loss
        
        # tell model we are evaluating
        model.eval()

        # re-run on training data to get current train loss for the epoch
        train_pred_coeff = model(train_X.float())
        train_loss = loss_function(train_pred_coeff, train_bottleneck_X, train_y, MEF_reg_weight, MDF_reg_weight).item()
        train_losses.append(train_loss)
#         del train_loss

        # run on val data to evaluate how we are doing
        val_pred_coeff = model(val_X.float())
        val_loss = loss_function(val_pred_coeff, val_bottleneck_X, val_y, MEF_reg_weight, MDF_reg_weight).item()
        val_losses.append(val_loss)
        val_r2 = get_r_squared(val_pred_coeff, val_bottleneck_X, val_y)
        val_mae = get_mean_abs_err(val_pred_coeff, val_bottleneck_X, val_y)
#         del val_loss
        
        if not final_point_only:
            val_r2_list = val_r2
            val_mae_list = val_mae
            # evaluate on last point in sequence only
            val_r2 = val_r2[-1] 
            val_mae = val_mae[-1]
            # update best seen at any point in sequence
            val_r2_any_point = max(val_r2_list)
            if val_r2_any_point > best_r2_any_point:
                best_r2_any_point = val_r2_any_point
                best_r2_any_point_list = val_r2_list
            val_mae_any_point = min(val_mae_list) 
            if val_mae_any_point < best_mae_any_point:
                best_mae_any_point = val_mae_any_point
                best_mae_any_point_list = val_mae_list

        # always keep best r2 and mae updated
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_r2_epoch = epoch
        if val_mae < best_mae:
            best_mae = val_mae
            best_mae_epoch = epoch

        # check if we should save best r2 model
        if val_r2 > best_r2_model["r2"]:
            # only continue with saving if this model has no invalids
            if sum(get_count_invalid_preds(val_pred_coeff))==0 and sum(get_count_invalid_preds(train_pred_coeff))==0:
                # update best-r2-model stats
                best_r2_model["r2"] = val_r2
                best_r2_model["epoch"] = epoch
                best_r2_model["mae"] = val_mae
                if not final_point_only:
                    best_r2_model["seq-wise-r2"] = val_r2_list
                    best_r2_model["seq-wise-mae"] = val_mae_list
                # delete prev best-r2-model unless best-mae-model still points to it
                if "save_path" in best_r2_model \
                and ("save_path" not in best_mae_model \
                or best_r2_model["save_path"] != best_mae_model["save_path"]):
                    Path(best_r2_model["save_path"]).unlink() # delete prev-best model
                # save model if above threshold
                if val_r2 > min_save_r2:
                    model_save_name = f"epoch={epoch},r2={val_r2:.4f},mae={int(val_mae)},Invalids=0.pth"
                    save_model_path = os.path.join(model_dir, model_save_name)
                    best_r2_model["save_path"] = save_model_path
                    torch.save(model.state_dict(), save_model_path)

        # check if we should save best mae model
        if val_mae < best_mae_model["mae"]:
            # if we already saved best-r2-model on this round, then that model is our best mae model
            if epoch in best_r2_model and best_r2_model["epoch"] == epoch:
                best_mae_model["r2"] = val_r2
                best_mae_model["epoch"] = epoch
                best_mae_model["mae"] = val_mae
                if not final_point_only:
                    best_mae_model["seq-wise-r2"] = val_r2_list
                    best_mae_model["seq-wise-mae"] = val_mae_list
                # delete prev best-mae-model
                if "save_path" in best_mae_model:
                    Path(best_mae_model["save_path"]).unlink() # delete prev-best model
                best_mae_model["save_path"] = best_r2_model["save_path"]
            # Otherwise need to check again if this model has no invalids
            elif sum(get_count_invalid_preds(val_pred_coeff))==0 and sum(get_count_invalid_preds(train_pred_coeff))==0:
                # update best-mae-model stats
                best_mae_model["r2"] = val_r2
                best_mae_model["epoch"] = epoch
                best_mae_model["mae"] = val_mae
                if not final_point_only:
                    best_mae_model["seq-wise-r2"] = val_r2_list
                    best_mae_model["seq-wise-mae"] = val_mae_list
                # delete prev best-mae-model unless best-r2-model still points to it
                if "save_path" in best_mae_model \
                and ("save_path" not in best_r2_model \
                or best_r2_model["save_path"] != best_mae_model["save_path"]):
                    Path(best_mae_model["save_path"]).unlink() # delete prev-best model
                # save if below threshold
                if val_mae < max_save_mae:
                    model_save_name = f"epoch={epoch},r2={val_r2:.4f},mae={int(val_mae)},Invalids=0.pth"
                    save_model_path = os.path.join(model_dir, model_save_name)
                    best_mae_model["save_path"] = save_model_path
                    torch.save(model.state_dict(), save_model_path)

        # print performance info every so often
        if verbose and epoch % print_freq == 0:
            invalid_train_MEFs, invalid_train_MDFs = get_count_invalid_preds(train_pred_coeff)
            invalid_val_MEFs, invalid_val_MDFs = get_count_invalid_preds(val_pred_coeff)
            train_r2 = get_r_squared(train_pred_coeff, train_bottleneck_X, train_y)
            if not final_point_only:
                # evaluate on last point in sequence only
                train_r2 = train_r2[-1] 
            print(f"[Epoch {epoch}]")
            print(f"\tTrain Set: Loss={train_loss:.3e}, R Squared={train_r2:.4f}, Invalid MEFs={invalid_train_MEFs}, Invalid MDFs={invalid_train_MDFs}")
            print(f"\tVal Set: Loss={val_loss:.3e}, R Squared={val_r2:.4f}, Invalid MEFs={invalid_val_MEFs}, Invalid MDFs={invalid_val_MDFs}")

        # stop if we aren't improving after 10k epochs
        if best_r2_epoch and epoch > 10000 + best_r2_epoch:
            if verbose:
                print("Early stopping as we haven't made an improvement on validation set in 10,000 epochs.")
            break
    
        del val_pred_coeff
        del train_pred_coeff
    
    plot_losses(train_losses, val_losses, model_dir, verbose)

    results_str = f"Best R Squared seen on epoch {best_r2_epoch}: {best_r2:.4f}"
    results_str += f"\nBest MAE seen on epoch {best_mae_epoch}: {best_mae:.2f}"
    
    results_str += f"\nBest-R2-model with 0 invalid coefficients predicted on train/val sets:"
    results_str += f"\n\tValidation R2: {best_r2_model['r2']:.4f}"
    results_str += f"\n\tValidation MAE: {best_r2_model['mae']:.2f}"
    if not final_point_only:
        r2_str_list = [f"{val:.4f}" for val in best_r2_model['seq-wise-r2']]
        results_str += f"\n\tValidation sequence-wise-R2: {r2_str_list}"
        mae_str_list = [f"{val:.2f}" for val in best_r2_model['seq-wise-mae']]
        results_str += f"\n\tValidation sequence-wise-MAE: {mae_str_list}"
    results_str += f"\n\tEpoch seen: {best_r2_model['epoch']}"
    if "save_path" not in best_r2_model:
        results_str += f"\n\tNo such model with R2 above {min_save_r2=} was encountered."
    else:
        results_str += f"\n\tModel file: {best_r2_model['save_path'].split('/')[-1]}"
    
    results_str += f"\nBest-MAE-model with 0 invalid coefficients predicted on train/val sets:"
    results_str += f"\n\tValidation R2: {best_mae_model['r2']:.4f}"
    results_str += f"\n\tValidation MAE: {best_mae_model['mae']:.2f}"
    if not final_point_only:
        r2_str_list = [f"{val:.4f}" for val in best_mae_model['seq-wise-r2']]
        results_str += f"\n\tValidation sequence-wise-R2: {r2_str_list}"
        mae_str_list = [f"{val:.2f}" for val in best_mae_model['seq-wise-mae']]
        results_str += f"\n\tValidation sequence-wise-MAE: {mae_str_list}"
    results_str += f"\n\tEpoch seen: {best_mae_model['epoch']}"
    if "save_path" not in best_mae_model:
        results_str += f"\n\tNo model with MAE below {max_save_mae=} was encountered."
    else:
        results_str += f"\n\tModel file: {best_mae_model['save_path'].split('/')[-1]}"
        
    if not final_point_only:
        results_str += f"\nSequence-wise-R2 with the highest R2 at any point in the sequence of {best_r2_any_point:.4f}:"
        r2_str_list = [f"{val:.4f}" for val in best_r2_any_point_list]
        results_str += f"\n\t{r2_str_list}"
        results_str += f"\nSequence-wise-MAE with the lowest MAE at any point in the sequence of {best_mae_any_point:.2f}:"
        mae_str_list = [f"{val:.2f}" for val in best_mae_any_point_list]
        results_str += f"\n\t{mae_str_list}"

    if verbose:
        print(results_str)
    with open(os.path.join(model_dir, "results.txt"), 'w+') as f:
        f.write(results_str)
        
    
    return (best_r2_model['r2'] if 'r2' in best_r2_model else None,
            best_mae_model['mae'] if 'mae' in best_mae_model else None)

## Old un-batched training function -- likely will remove this.
# def train_model_with_params(train_X, val_X, train_bottleneck_X, val_bottleneck_X, train_y, val_y,  # data
#                             input_size, hidden_size, output_size, final_point_only, num_layers, dropout,  # model settings
#                             learning_rate, weight_decay,  # optimizer settings
#                             loss_function, MEF_reg_weight, MDF_reg_weight,  # loss function settings
#                             model_dir_prefix=None, epochs=10000, print_freq=1000, min_save_r2=.87):  # train process settings

#     # if predicting final point in sequence only, keep only those corresponding points
#     # in the bottleneck_X and y arrays
#     if final_point_only:
#         train_bottleneck_X = torch.unsqueeze(train_bottleneck_X[:,-1,:], 1)
#         val_bottleneck_X = torch.unsqueeze(val_bottleneck_X[:,-1,:], 1)
#         train_y = torch.unsqueeze(train_y[:,-1], 1)
#         val_y = torch.unsqueeze(val_y[:,-1], 1)

#     # create model and optimizer
#     model = LSTM(input_size, hidden_size, output_size, final_point_only, num_layers, dropout)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Set up folder where model and model info will be saved
#     model_dir = model_save_dir
#     if model_dir_prefix:
#         model_dir += f"/{model_dir_prefix}"
#     model_dir += f"/{datetime.now().strftime('%Y_%m_%d-%I:%M:%S_%p')}"
#     Path(model_dir).mkdir(parents=True, exist_ok=True)

#     # log experiment settings
#     settings_str = "Model Settings:"
#     settings_str += f"\n\t{input_size=}\n\t{hidden_size=}\n\t{output_size=}\n\t{final_point_only=}\n\t{num_layers=}\n\t{dropout=}"
#     settings_str += "\nOptimizer Settings:"
#     settings_str += f"\n\t{learning_rate=}\n\t{weight_decay=}"
#     settings_str += "\nLoss Function Settings:"
#     settings_str += f"\n\t{loss_function=}\n\t{MEF_reg_weight=}\n\t{MDF_reg_weight=}"
#     settings_str += "\nTrain Process Settings:"
#     settings_str += f"\n\t{epochs=}\n\t{min_save_r2=}"
#     settings_str += f"\nFeatures: {', '.join(feature_cols)}"
#     print(settings_str)
#     with open(f"{model_dir}/experiment_settings.txt", 'w+') as f:
#         f.write(settings_str)

#     best_r2 = -np.inf 
#     best_epoch = None
#     best_model_mae = -np.inf
#     save_model_path = None
#     last_save_epoch = None
#     last_save_r2 = -np.inf
    
#     train_losses = []
#     val_losses = []
#     for epoch in tqdm(range(epochs)):
#         # tell model we are training
#         model.train()
#         train_pred_coeff = model(train_X.float())
#         train_loss = loss_function(train_pred_coeff, train_bottleneck_X, train_y, MEF_reg_weight, MDF_reg_weight)
#         train_losses.append(train_loss.item())
        
#         # tell model we are evaluating
#         model.eval()
#         val_pred_coeff = model(val_X.float())
#         val_loss = loss_function(val_pred_coeff, val_bottleneck_X, val_y, MEF_reg_weight, MDF_reg_weight)
#         val_losses.append(val_loss.item())
#         val_r2 = get_r_squared(val_pred_coeff, val_bottleneck_X, val_y)

#         # always keep best r2 updated
#         if val_r2 > best_r2:
#             best_r2 = val_r2
#             best_model_mae = get_mean_abs_err(val_pred_coeff, val_bottleneck_X, val_y)
#             best_epoch = epoch
#         # check if we should save... we need good enough r2 and no invalids
#         if val_r2 > max(last_save_r2, min_save_r2):
#             if sum(get_count_invalid_preds(val_pred_coeff))==0:
#                 # also check training invalids... Let's recompute with eval mode
#                 model.eval()
#                 eval_mode_train_preds=model(train_X.float()).cpu()
#                 if sum(get_count_invalid_preds(eval_mode_train_preds))==0:
#                     if save_model_path:
#                         Path(save_model_path).unlink() # delete prev-best model
#                     model_save_name = f"epoch={epoch},r2={val_r2:.4f},Invalids=0.pth"
#                     save_model_path = f"{model_dir}/{model_save_name}"
#                     torch.save(model.state_dict(), save_model_path)
#                     last_save_epoch = epoch
#                     last_save_r2 = val_r2

#         if epoch % print_freq == 0:
#             invalid_train_MEFs, invalid_train_MDFs = get_count_invalid_preds(train_pred_coeff)
#             invalid_val_MEFs, invalid_val_MDFs = get_count_invalid_preds(val_pred_coeff)
#             train_r2 = get_r_squared(train_pred_coeff, train_bottleneck_X, train_y)
#             print(f"[Epoch {epoch}]")
#             print(f"\tTrain Set: Loss={train_loss.item():.3e}, R Squared={train_r2:.4f}, Invalid MEFs={invalid_train_MEFs}, Invalid MDFs={invalid_train_MDFs}")
#             print(f"\tVal Set: Loss={val_loss.item():.3e}, R Squared={val_r2:.4f}, Invalid MEFs={invalid_val_MEFs}, Invalid MDFs={invalid_val_MDFs}")

#         model.zero_grad()
#         train_loss.backward()
#         optimizer.step()

#         # stop if we aren't improving after 10k epochs
#         if best_epoch and epoch > 10000 + best_epoch:
#             print("Early stopping as we haven't made an improvement on validation set in 10,000 epochs.")
#             break
    
#     print_results(train_losses, train_pred_coeff, train_bottleneck_X, train_y,
#                   val_losses, val_pred_coeff, val_bottleneck_X, val_y, model_dir)
#     print(f"best R Squared seen on epoch {best_epoch}: {best_r2:.4f}")
#     if save_model_path:
#         print(f"best R Squared with no invalids predicted in train/val seen on epoch {last_save_epoch}: {last_save_r2:.4f}")
#         with open(f"{save_model_path[:-4]}.results.txt", 'w+') as f:
#             f.write("Val Set:")
#             f.write(f"\tMean Absolute Error={best_model_mae:.2f}")
#             f.write(f"\tR Squared={best_r2:.4f}")
#     else:
#         print(f"No model was saved because no model that had no train/val invalids reached {min_save_r2} validation R2.")
        
#     return save_model_path

