import numpy as np 
import torch, os, argparse, random
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
# from rich import print
basedir = os.path.abspath(os.path.dirname(__file__))
os.chdir(basedir)
os.makedirs("topk_indices", exist_ok=True)
os.makedirs("cooc_results", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("bias_figs", exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
from model import CreaTDA, CreaTDA_og
import matplotlib.pyplot as plt
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26, help="global random seed")
    parser.add_argument("--d", default=1024, type=int, help="embedding dimension d")
    parser.add_argument("--k",default=512, type=int, help="dimension of project matrices k")
    parser.add_argument("--l2-factor",default = 1.0, type=float, help="weight of l2 regularization")
    parser.add_argument("--model",default = "CreaTDA", type=str, help="model type", choices=['CreaTDA_og', 'CreaTDA', 'GTN', 'DTINet', 'RGCN', 'HGT'])
    parser.add_argument("--device", choices=[-1,0,1,2,3], default=0, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--top", default=200, type=int, help="number of top indices")
    args = parser.parse_args()
    return args

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return torch.Tensor(new_matrix)

def predict(args, TDAtrain):
    set_seed(args)
    protein_disease = np.zeros((num_protein, num_disease))
    mask = np.zeros((num_protein, num_disease))
    for ele in TDAtrain:
        protein_disease[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    disease_protein = protein_disease.T

    disease_protein_normalize = row_normalize(disease_protein,False).to(device)
    protein_disease_normalize = row_normalize(protein_disease,False).to(device)
    protein_disease = torch.Tensor(protein_disease).to(device)
    mask = torch.Tensor(mask).to(device)
    model_path = 'models/CreaTDA_{}_retrain.pth'.format(args.model)
    if args.model == 'CreaTDA_og':
        model = CreaTDA_og(args, num_drug, num_disease, num_protein, num_sideeffect)
    elif args.model == 'CreaTDA':
        model = CreaTDA(args, num_drug, num_disease, num_protein, num_sideeffect)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    
    ground_truth_train = [ele[2] for ele in TDAtrain]
    best_train_aupr = 0
    best_train_auc = 0

    model.eval()
    if args.model == 'CreaTDA_og':
        tloss, tdaloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, 
                                    drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, 
                                    protein_disease_normalize, disease_drug_normalize, disease_protein_normalize, 
                                    sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, 
                                    drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, 
                                    protein_sequence, protein_disease, drug_protein, mask)
    elif args.model == 'CreaTDA':
        tloss, tdaloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, 
                                        drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, 
                                        protein_disease_normalize, disease_drug_normalize, disease_protein_normalize, 
                                        sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, 
                                        drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, 
                                        protein_sequence, protein_disease, drug_protein, mask, drug_protein_co_weight, drug_disease_co_weight, protein_disease_co_weight, 
                                        drug_protein_co_label, drug_disease_co_label, protein_disease_co_label)
    pred_list_train = [results[ele[0],ele[1]] for ele in TDAtrain]
    train_auc = roc_auc_score(ground_truth_train, pred_list_train)
    train_aupr = average_precision_score(ground_truth_train, pred_list_train)
    print ('auc aupr', train_auc, train_aupr)
    
    return results

def get_topk(args, output, tda):
    # 1. 3-sigma rule for significance for each column (disease)
    output_mean = np.mean(output, axis=0) # mean over proteins
    output_std = np.std(output, axis=0)
    sig_thresh = output_mean + 2 * output_std
    # print(output_mean.shape, output_std.shape, tda.shape)
    list_of_indices = []
    for i in range(output.shape[1]):
        column = output[:, i]
        sig_indices = np.where(column >= sig_thresh[i])[0] # array of indices
        if len(sig_indices) > 0:
            list_of_indices.extend([(idx, i) for idx in sig_indices])
    new_output = np.zeros_like(output)
    for tup in list_of_indices:
        new_output[tup[0], tup[1]] = output[tup[0], tup[1]]
    new_output = new_output * (1-tda)
    tensor_output = torch.tensor(new_output.flatten())
    top_k = torch.topk(tensor_output, args.top).indices.numpy()
    real_indices = np.unravel_index(top_k, tda.shape)
    # print(real_indices[0])
    real_indices = np.array(real_indices).T
    return real_indices

def get_topk_cooc(args, output, tda, cooc):
    new_output = output * (1-tda)
    # new_output = np.where(cooc >= 0, np.zeros_like(new_output), new_output)
    true_output = np.where(cooc<=0, np.zeros_like(new_output), new_output).flatten()
    false_output = np.where(cooc>0, np.zeros_like(new_output), new_output).flatten()

    true_output = torch.tensor(true_output[np.nonzero(true_output)[0]])
    false_output = torch.tensor(false_output[np.nonzero(false_output)[0]])

    true_output_top = true_output[list(torch.topk(true_output, args.top).indices.numpy())].numpy()
    false_output_top = false_output[list(torch.topk(false_output, args.top).indices.numpy())].numpy()

    return true_output_top, false_output_top

def get_corr_cooc(output, indices, cooc):
    output_vec = np.array([output[entry[0]][entry[1]] for entry in indices])
    cooc_vec = np.array([cooc[entry[0]][entry[1]] for entry in indices])
    corr, pval = spearmanr(output_vec, cooc_vec)
    return corr, pval

def plot_bias_figs(output, tda, indices):
    # plot max scores
    num_known_diseases = np.sum(tda, axis=1)   # num_proteins

    max_score_proteins = np.max(output, axis=1)

    protein_score_association_corr = spearmanr(num_known_diseases, max_score_proteins)


    fig, ax = plt.subplots()
    ax.scatter(num_known_diseases, max_score_proteins, c='dodgerblue',s=1)
    fig.tight_layout()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(11)
    fig.savefig("bias_figs/max_scores_{}.png".format(args.model), bbox_inches = 'tight', pad_inches = 0.1, dpi=400)
    plt.close(fig)
    print("{}: protein max scores and known associated diseases.\n correlation: {:.3f} p-value: {:.3f}".format(args.model, protein_score_association_corr[0], protein_score_association_corr[1]))

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    network_path = '../data/'
    tda_o = np.loadtxt(network_path+'mat_protein_disease.txt')
    # TODO Here
    
    if not (args.model == 'DTINet' or args.model == 'GTN' or args.model == 'HGT' or args.model == 'RGCN'):
        print('loading networks ...')
        drug_drug = np.loadtxt(network_path+'mat_drug_drug.txt')
        true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
        drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')
        drug_chemical=drug_chemical[:true_drug,:true_drug]
        drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt')
        drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')
        disease_drug = drug_disease.T
        sideeffect_drug = drug_sideeffect.T

        protein_protein = np.loadtxt(network_path+'mat_protein_protein.txt')
        protein_sequence = np.loadtxt(network_path+'Similarity_Matrix_Proteins.txt')
        protein_drug = np.loadtxt(network_path+'mat_protein_drug.txt')
        drug_protein = protein_drug.T

        if args.model == 'CreaTDA':
            drug_protein_co_weight = torch.Tensor(np.load(network_path+"drug_protein_co_weight.npy", allow_pickle=True)).to(device)
            drug_disease_co_weight = torch.Tensor(np.load(network_path+"drug_disease_co_weight.npy", allow_pickle=True)).to(device)
            protein_disease_co_weight = torch.Tensor(np.load(network_path+"protein_disease_co_weight.npy", allow_pickle=True)).to(device)
            drug_protein_co_label = torch.Tensor(np.load(network_path+"drug_protein_co_label.npy", allow_pickle=True)).to(device)
            drug_disease_co_label = torch.Tensor(np.load(network_path+"drug_disease_co_label.npy", allow_pickle=True)).to(device)
            protein_disease_co_label = torch.Tensor(np.load(network_path+"protein_disease_co_label.npy", allow_pickle=True)).to(device)

        print('normalize network for mean pooling aggregation')
        drug_drug_normalize = row_normalize(drug_drug,True).to(device)
        drug_chemical_normalize = row_normalize(drug_chemical,True).to(device)
        drug_protein_normalize = row_normalize(drug_protein,False).to(device)
        drug_disease_normalize = row_normalize(drug_disease,False).to(device)
        drug_sideeffect_normalize = row_normalize(drug_sideeffect,False).to(device)

        protein_protein_normalize = row_normalize(protein_protein,True).to(device)
        protein_sequence_normalize = row_normalize(protein_sequence,True).to(device)
        protein_drug_normalize = row_normalize(protein_drug,False).to(device)

        disease_drug_normalize = row_normalize(disease_drug,False).to(device)
        
        sideeffect_drug_normalize = row_normalize(sideeffect_drug,False).to(device)

        #define computation graph
        num_drug = len(drug_drug_normalize)
        num_protein = len(protein_protein_normalize)
        num_disease = len(disease_drug_normalize)
        num_sideeffect = len(sideeffect_drug_normalize)

        drug_drug = torch.Tensor(drug_drug).to(device)
        drug_chemical = torch.Tensor(drug_chemical).to(device)
        drug_disease = torch.Tensor(drug_disease).to(device)
        drug_sideeffect = torch.Tensor(drug_sideeffect).to(device)
        protein_protein = torch.Tensor(protein_protein).to(device)
        protein_sequence = torch.Tensor(protein_sequence).to(device)
        protein_drug = torch.Tensor(protein_drug).to(device)
        drug_protein = torch.Tensor(drug_protein).to(device)

        #prepare drug_protein and mask
        
        whole_positive_index = []
        whole_negative_index = []
        for i in range(np.shape(tda_o)[0]):
            for j in range(np.shape(tda_o)[1]):
                if int(tda_o[i][j]) == 1:
                    whole_positive_index.append([i,j])
                elif int(tda_o[i][j]) == 0:
                    whole_negative_index.append([i,j])
        negative_sample_index = np.arange(len(whole_negative_index))
        data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        print ('Predicting ...')
        unfiltered_output = predict(args=args, TDAtrain=data_set)
        np.save("topk_indices/output_raw_{}.npy".format(args.model), unfiltered_output, allow_pickle=True)
        
    # TODO Here
    cooc = np.load("../../literature_data/output/protein_disease.npy", allow_pickle=True)
    unfiltered_output = np.load("topk_indices/output_raw_{}.npy".format(args.model), allow_pickle=True)    
    output = np.where(tda_o != 1, unfiltered_output, 0)  # only those positions on which ground-truth=0
    
    # compute top indices and correlation
    indices = get_topk(args, unfiltered_output, tda_o)
    corr, pval = get_corr_cooc(unfiltered_output, indices, cooc)
    print("correlation between top {} co-occurrences and output is {:.3f} with p-value {:.3f}".format(args.top, corr, pval))

    # TODO get bias figs
    plot_bias_figs(output, tda_o, indices)

    coocs = np.array([cooc[item[0]][item[1]] for item in indices])
    np.save("cooc_results/cooc_{}.npy".format(args.model), coocs)
    
    # write to file
    os.chdir(os.path.join(basedir, "../data"))
    protein_dict = np.load("protein_dict.npy", allow_pickle=True).item() 
    result_path = os.path.abspath("../src/results/{}_results.tsv".format(args.model))
    with open("protein.txt", "r") as protein_txt, open("disease.txt", "r") as disease_txt, open(result_path, "w+") as results:
        proteins = protein_txt.readlines() 
        diseases = disease_txt.readlines()
        results.write('protein_idx'+'\t'+"disease_idx"+'\t'+'disease'+'\t'+'protein_number'+'\t'+'protein_name'+'\t'+'output'+'\t'+'co'+'\n')
        for i in range(len(indices)):
            protein_idx = indices[i][0]
            disease_idx = indices[i][1]
            # print(disease_idx)
            disease = diseases[disease_idx].strip()
            protein_number = proteins[protein_idx].strip()
            protein_name = protein_dict[protein_number]
            # print(output.shape)
            pred = output[protein_idx, disease_idx]
            co = cooc[protein_idx, disease_idx]
            # results.write(str(protein_idx)+'\t'+str(disease_idx)+'\t'+ disease + '\t'+ protein_number+'\t'+protein_name+'\t'+ str(pred) + str(label) +'\n')
            results.write('{}\t{}\t{}\t{}\t{}\t{:.3f}\t{}\n'.format(protein_idx, disease_idx, disease, protein_number, protein_name, pred, co))
