# Purpose:
# 1. Plot tree


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Plot_Tree import Plot_Tree

########################################################################################################################
# Set up
########################################################################################################################
tree_type = "level_tree"         # "original_tree", "level_tree", "remained_tree"
input_folder = Desktop + '123/'
nrn_lst = neuron_dict["all"]
# remove_method = "leaf"             # None or "leaf"
# target_level = 5                   # None or int
remove_method = None  # None or "leaf"
target_level = None
with_mix_point = True
file_type = 'pdf'                  # "jpeg" or "pdf" (more see: https://www.graphviz.org/doc/info/output.html)
overwrite = True




########################################################################################################################
# Main Code
########################################################################################################################
if __name__ == "__main__":
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for nrn in bar(nrn_lst):
        tree = Plot_Tree(tree_type, input_folder, nrn, remove_method, target_level, with_mix_point, file_type, overwrite=overwrite)
        tree.plot_tree()
    time.sleep(0.01)


########################################################################################################################
# End of Code
########################################################################################################################
