# Purpose:
# from csv
# 1. Plot original tree
# 2. Plot level tree
# 3. Plot remained level tree


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Plot_Result_Tree import Plot_Result_Tree


########################################################################################################################
# Set up
########################################################################################################################

model_lst = ["gbdt", "svm", "rf", "xgb"]
# input_folder = './data/'
input_folder = Desktop + '123/'
train_nrn_type = ["normal"]  # "all", "normal", or "multiple"
test_nrn_type = ["normal"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
feature_type = 'f4'
with_replacement = 50
with_mix_point = False
file_type = 'pdf'





########################################################################################################################
# Main Code
########################################################################################################################

bar = progressbar.ProgressBar()
time.sleep(0.01)
for model in bar(model_lst):

    tree = Plot_Result_Tree(model, input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, feature_type, with_replacement, with_mix_point, file_type)

    tree.plot_tree()

time.sleep(0.01)
    
########################################################################################################################
# End of Code
########################################################################################################################
