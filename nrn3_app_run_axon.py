# Purpose:
# 1. Classify and evaluate axon


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Classify_Axon import Classify_Axon

########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", "small", "multiple", or "axonNear", "normalNew", "multipleNew"
test_nrn_type = ["all"]  # "all", "normal", "small", "multiple", or "axonNear"
remove_method = "leaf"
target_level = 5
feature_type='af9'
with_replacement = 50
train_with_only_axon_dendrite = False
overwrite = False
seed = 123

# overwrite = True

########################################################################################################################
# Main Code
########################################################################################################################
if __name__ == '__main__':
    ax0 = Classify_Axon(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, feature_type, with_replacement, train_with_only_axon_dendrite, seed, overwrite=overwrite)
    ax0 = ax0.classify_data()
    ax0.evaluation_info()



########################################################################################################################
# End of Code
########################################################################################################################
