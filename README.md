# CreaTDA

## Requirements

* pytorch (tested on version 1.10.0+cu102)

* numpy (test on version 1.22.0)
* sklearn (tested on version 1.0.1)
* matplotlib (tested on version 3.5.1)
* argparse, rich(optional)

## Quick start

To reproduce our results:

1. Extract data.tgz at the project root folder
2. Go to src/ folder
2. Run <code>generate_co.py</code> to generate co-occurrence-dependent labels and weights.
3. Run <code>pytorch_CreaTDA_cv.py</code> to reproduce cross validation results. Command line arguments are: 

```shell
--seed: global random seed. default: 26
--d: embedding dimension d. default: 1024
--n: global gradient norm to be clipped. default: 1
--k: dimension of project matrices k. default: 512
--model: model type, choices: CreaTDA_og, CreaTDA. default: CreaTDA
--l2-factor: weight of l2 regularization. default: 1
--lr: learning rate. default: 1e-3
--weight-decay: weight decay of optimizer. default: 0
--num-steps: number of training steps. default: 3000
--device: device number (-1 for cpu). default: 0
--n-folds: number of folds for cross validation. default: 5
--round: number of rounds of cross validation. default: 10
--test-size: portion of validation data w.r.t. trainval-set. default: .1
--mask: masking scheme, choices: random, tda_disease. default: random
```

The **point-wise cross validation scheme** corresponds to <code>--mask random</code> while the cluster-wise cross validation scheme corresponds to <code>--mask tda_disease</code>.

3. Run <code>pytorch_CreaTDA_retrain.py</code> to retrain the model on the full HN and save the model. Command line arguments are:

```shell
--seed: global random seed. default: 26
--d: embedding dimension d. default: 1024
--n: global gradient norm to be clipped. default: 1
--k: dimension of project matrices k. default: 512
--model: model type, choices: CreaTDA_og, CreaTDA. default: CreaTDA
--l2-factor: weight of l2 regularization. default: 1
--lr: learning rate. default: 1e-3
--weight-decay: weight decay of optimizer. default: 0
--num-steps: number of training steps. default: 3000
--device: device number (-1 for cpu). default: 0
```

4. Run <code>get_top_pred.py</code> to retrieve figures, statistics, and predictions. Command line arguments are: 

```shell
--seed: global random seed. default: 26
--d: embedding dimension d. default: 1024
--n: global gradient norm to be clipped, default: 1
--l2-factor: weight of l2 regularization. default: 1
--model: model type, choices: CreaTDA_og, CreaTDA, GTN, DTINet, RGCN, HGT. default: CreaTDA
--device: device number (-1 for cpu), default: 0
```

## Data description

### Individual biological networks 

These data are in the data/ folder

* drug.txt: list of drug names.

* protein.txt: list of protein names.

* disease.txt: list of disease names.

* se.txt: list of side effect names.
* drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
* protein_dict_map: a complete ID mapping between protein names and UniProt ID.
* mat_drug_se.txt : Drug-SideEffect association matrix.
* mat_protein_protein.txt : Protein-Protein interaction matrix.
* mat_drug_drug.txt : Drug-Drug interaction matrix.
* mat_protein_disease.txt : Protein-Disease association matrix.
* mat_drug_disease.txt : Drug-Disease association matrix.
* mat_protein_drug.txt : Protein-Drug interaction matrix.
* mat_drug_protein.txt : Drug-Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs (\[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.
* mat_drug_protein_homo_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
* mat_drug_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_sideeffect.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_disease.txt: Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_unique: Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 
* mat_compound_protein_bindingaffinity.txt: Compound-Protein binding affinity matrix (measured by negative logarithm of *_**Ki**_*).

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files. These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.

### Co-occurrence counts

These data are in the literature_data/ folder

* protein_disease_new.txt: co-occurrence counts between proteins and diseases.
* protein_drug_new.txt: co-occurrence counts between proteins and drugs.
* drug_disease_new.txt: co-occurrence counts between drugs and diseases.

## Contacts

If you have any questions or comments, please feel free to email Chang Liu (liu-chan19[at]mails[dot]tsinghua[dot]edu[dot]cn) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).