# Purpose:
# from csv
# 1. Plot original tree
# 2. Plot level tree
# 3. Plot remained level tree


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Data_Cleaner import Data_Cleaner


########################################################################################################################
# Set up
########################################################################################################################
tree_type = "reduce_on_level"         # "original_tree", "level_tree", "remained_tree"
input_folder = Desktop + '123/'


# nrn_lst = neuron_dict["all"]
nrn_lst = ["VGlut-F-800100","Gad1-F-800046","Cha-F-500046","Cha-F-100117","Gad1-F-600003","Gad1-F-400400","5-HT1B-F-500013","Cha-F-700121","Trh-F-300113","Gad1-F-100602","Cha-F-700121","VGlut-F-200012","VGlut-F-900093","VGlut-F-000600","Trh-F-400043"]
# nrn_lst = ["5-HT1B-F-500013"]     # test
# nrn_lst = ["Cha-F-500046"]  # bad
# nrn_lst = ["Gad1-F-600003"]  # bad
# nrn_lst = ["VGlut-F-000600"]  # good
# nrn_lst = ["VGlut-F-200012"]
overwrite = True

remove_method = "leaf"             # None or "leaf"
target_level = 5                   # None or int
with_mix_point = False
file_type = 'jpg'




########################################################################################################################
# Main Code
########################################################################################################################

class Plot_Tree():

    def __init__(self,
                 tree_type,  # "original_tree", "level_tree", "remained_tree"
                 input_folder,
                 nrn_name,
                 remove_method=None,
                 target_level=None,
                 with_mix_point=False,
                 file_type='pdf',
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite = False
                 ):


        self.input_folder = input_folder
        self.nrn_name = nrn_name
        self.tree_type = tree_type
        self.remove_method = remove_method
        self.target_level = target_level
        self.with_mix_point = with_mix_point
        self.file_type = file_type
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        self.file_size = '20, 20'


        # fname
        if self.tree_type == "original_tree":
            _fname0 = self.nrn_name

            # _withMixPoint
            if self.with_mix_point:
                _fname0 = "_".join([_fname0, "mx"])


        elif self.tree_type == "level_tree":
            if all([self.remove_method is None, self.target_level is None]):
                _fname0 = self.nrn_name

                # _withMixPoint
                if self.with_mix_point:
                    _fname0 = "_".join([_fname0, "mx"])

            elif all([type(self.remove_method) is str, type(self.target_level) is int]):
                # _methodLevel
                _methodLevel = remove_method + str(target_level)
                _fname0 = "_".join([self.nrn_name, _methodLevel])

                # _withMixPoint
                if self.with_mix_point:
                    _fname0 = "_".join([_fname0, "mx"])



        elif self.tree_type == "reduce_on_level":
            if all([type(self.remove_method) is str, type(self.target_level) is int]):
                # _methodLevel
                _methodLevel = remove_method + str(target_level)

                # _levelDescend
                n1 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
                n1 = n1.load_data()
                level0 = n1.df_level_descend["level"].max()
                level1 = n1.df_level_descend["level"].min()
                _levelDescend = "".join([str(level0), "to", str(level1)])
                _fname0 = "_".join([self.nrn_name, _methodLevel, _levelDescend])

                # _withMixPoint
                if self.with_mix_point:
                    _fname0 = "_".join([_fname0, "mx"])

            else:

                sys.exit("\n If tree_type = 'remained_tree', then remove_method and target_level should not be None!")



        elif self.tree_type == "remained_tree":
            if all([type(self.remove_method) is str, type(self.target_level) is int]):
                # _methodLevel
                _methodLevel = remove_method + str(target_level)

                # _levelDescend
                n1 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
                n1 = n1.load_data()
                level0 = n1.df_level_descend["level"].max()
                level1 = n1.df_level_descend["level"].min()
                _levelDescend = "".join([str(level0), "to", str(level1)])
                _fname0 = "_".join([self.nrn_name, _methodLevel, _levelDescend])

                # _withMixPoint
                if self.with_mix_point:
                    _fname0 = "_".join([_fname0, "mx"])
                
            else:
                sys.exit("\n If tree_type = 'remained_tree', then remove_method and target_level should not be None!")

        self.save_path = input_folder + "nrn_plot/" + self.tree_type + "/"
        self.output_name = _fname0
        self.fname = self.save_path + self.output_name + ".gv." + self.file_type


        return


    def plot_tree(self):
        if all([self._is_ready(), not self.overwrite]):
            # print("The plot of ", self.tree_type, self.output_name, "exist.")
            pass
        else:
            self._plot_data_from_Data_Cleaner()
        return


    def _is_ready(self):
        if os.path.exists(self.fname):
            return True
        else:
            return False


    def _plot_data_from_Data_Cleaner(self):
        ### Read in original df & reduced df from Data_cleaner & polarity dict
        n0 = Data_Cleaner(self.input_folder, self.nrn_name)
        n0 = n0.load_data()
        df0 = n0.df
        df_dis0 = n0.df_dis
        polarity_dict0 = n0.polarity_dict

        n1 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
        n1 = n1.load_data()
        df1 = n1.df
        df_dis1 = n1.df_dis
        polarity_dict1 = n1.polarity_dict


        ### Prepare order list
        order_lst0 = df0['ID'].tolist()
        order_lst1 = df1['ID'].tolist()


        # remained_tree's polarity
        if self.tree_type == "remained_tree":
            polarity_dict0["mix"] = polarity_dict1["mix"]


        # with_mix_point
        if not self.with_mix_point:
            polarity_dict0['mix'] = []
            polarity_dict1['mix'] = []



        ### Plot
        if self.tree_type == "original_tree":
            neuron_plot_relation_tree(df0, self.child_col, self.parent_col, polarity_dict=polarity_dict0, node_order_lst=order_lst0, save_path=self.save_path, filename=self.output_name, subname=False, file_type=self.file_type, fig_size=self.file_size)


        elif self.tree_type == "level_tree":
            # original level tree
            if all([self.remove_method is None, self.target_level is None]):
                neuron_plot_relation_tree(df_dis0, 'descendant', 'ancestor', 'len', polarity_dict0, node_order_lst=order_lst0, save_path=self.save_path, filename=self.output_name, subname=False, file_type=self.file_type, fig_size=self.file_size)

            # reduced level tree
            elif all([type(self.remove_method) is str, type(self.target_level) is int]):
                neuron_plot_relation_tree(df_dis1, 'descendant', 'ancestor', 'len', polarity_dict1, node_order_lst=order_lst1, save_path=self.save_path, filename=self.output_name, subname=False, file_type=self.file_type, fig_size=self.file_size)


        elif self.tree_type == "reduce_on_level":
            if all([type(self.remove_method) is str, type(self.target_level) is int]):
                neuron_plot_reduce_on_level_tree(df_dis0, df_dis1, 'descendant', 'ancestor', 'len', polarity_dict0, node_order_lst=order_lst0, save_path=self.save_path, filename=self.output_name, subname=False, file_type=self.file_type, fig_size=self.file_size)

            else:
                sys.exit("\n If tree_type = 'remained_tree', then remove_method and target_level should not be None!")


        elif self.tree_type == "remained_tree":
            if all([type(self.remove_method) is str, type(self.target_level) is int]):
                neuron_plot_remained_tree(df_dis0, df_dis1, 'descendant', 'ancestor', 'len', polarity_dict0, node_order_lst=order_lst0, save_path=self.save_path, filename=self.output_name, subname=False, file_type=self.file_type, fig_size=self.file_size)

            else:
                sys.exit("\n If tree_type = 'remained_tree', then remove_method and target_level should not be None!")


        return


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
