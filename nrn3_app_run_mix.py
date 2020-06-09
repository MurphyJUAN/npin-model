# Purpose:
# 1. Create train & test set
# 2. Run model & Evaluate model
# 3. Outputs:
#       result_mix_leaf5_normal.pkl: train=test=normal, sampling the test & train set w/o replacement.
#       result_mix_leaf5_replace20_normal.pkl: train=test=normal, sampling w/ replacement for 20 times.
#       result_mix_leaf5_normal&small.pkl: train=normal, test=small, no sampling.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Classify_Mix import Classify_Mix



########################################################################################################################
# Set up
########################################################################################################################
# input_folder = './data/'
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
ax_feature_type = 'af8'
mx_feature_type = 'mf14'
with_replacement = 50
overwrite = False
seed=123


########################################################################################################################
# Main Code
########################################################################################################################


for af in ['af4', 'af8', 'af9']:
    for mf in ['mf4', 'mf8', 'mf9', 'mf10', 'mf14', 'mf15']:
        mx0 = Classify_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, af, mf, with_replacement, seed, overwrite=overwrite)

        mx0 = mx0.classify_data()

        mx0.evaluation_info()
        print("===========================================================================================")

# mx0 = Classify_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, mx_feature_type, with_replacement, seed, overwrite=overwrite)
#
# mx0 = mx0.classify_data()
#
# mx0.evaluation_info()




########################################################################################################################
