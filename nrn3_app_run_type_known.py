# Purpose:
# 1. Classify and evaluate type


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Classify_Type_Known import Classify_Type_Known


########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
odd_nrn_type = ["multiple", "axonNear"]  # "all", "normal", "small", "multiple", or "axon_near"
remove_method = "leaf"
target_level = 5
with_replacement = True
seed = 123


########################################################################################################################
# Main Code
########################################################################################################################
if __name__ == '__main__':
    oversampling = False  # without synthesize
    tp0 = Classify_Type_Known(input_folder, odd_nrn_type, remove_method, target_level, with_replacement, oversampling, seed)
    tp0 = tp0.classify_data()
    tp0.evaluation_info()



    oversampling = True  # without synthesize
    tp0 = Classify_Type_Known(input_folder, odd_nrn_type, remove_method, target_level, with_replacement, oversampling, seed)
    tp0 = tp0.classify_data()
    tp0.evaluation_info()




########################################################################################################################
# End of Code
########################################################################################################################
