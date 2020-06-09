# Py_Polarity_package

## Apps: 1. run_axon  2. run_mix  3. plot_tree  4. plot_result_tree

## Folder definition:
### main folder: "./data"
### original data: "./data/nrn_original" (.swc)
### cleaned data: "./data/nrn_cleaned"
### model results: "./data/nrn_result"
### plots: "./data/nrn_plot"


## 1. run_axon
### classify_data(): classify data
### evaluation_info(): show result
### Flow: Data_Cleaner -> Create_Axon_Feature -> Prepared_Axon -> Classify_Axon

## 2. run_mix
### classify_data(): classify data
### evaluation_info(): show result
### Flow: Classify_Axon & Create_Axon_Feature -> Create_Mix_Feature -> Prepare_Mix -> Classify_Mix

## 3. plot_tree
### plot_tree(): plot tree
### Flow: Data_Cleaner -> Plot_Tree

## 4. plot_result_tree
### plot_tree(): plot tree
### Flow: Data_Cleaner & Classify_Axon & Classify_Mix -> Plot_Result_Tree

