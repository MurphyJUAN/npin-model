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
from nrn3_class_Classify_Axon import Classify_Axon
from nrn3_class_Classify_Mix import Classify_Mix


########################################################################################################################
# Set up
########################################################################################################################

# model_lst = ["gbdt", "svm", "rf", "xgb"]
model_lst = ["gbdt", "rf", "xgb"]
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
ax_feature_type = 'af8'
mx_feature_type = 'mf14'
with_replacement = 50
with_mix_point = False
file_type = 'pdf'





########################################################################################################################
# Main Code
########################################################################################################################

class Plot_Result_Tree():

    def __init__(self,
                 model,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 ax_feature_type = 'af4',
                 mx_feature_type = None,
                 sample_with_replacement=None,
                 with_mix_point=True,
                 file_type='pdf',
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 ):

        self.model = model
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.ax_feature_type = ax_feature_type
        self.mx_feature_type = mx_feature_type
        self.with_replacement = sample_with_replacement
        self.with_mix_point = with_mix_point
        self.file_type = file_type
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col

        self.file_size = '20, 20'





        return


    def plot_tree(self):
        self._plot_data()
        return

    # def plot_tree(self):
    #     if all([self._is_ready(), not self.overwrite]):
    #         # print("The plot of ", self.tree_type, self.output_name, "exist.")
    #         pass
    #     else:
    #         self._plot_data_from_Data_Cleaner()
    #     return

    # def _is_ready(self):
    #     if os.path.exists(self.fname):
    #         return True
    #     else:
    #         return False


    def _plot_data(self):
        ### Read in original df & reduced df from Data_cleaner & polarity dict
        ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement)
        ax0 = ax0.classify_data()

        df_axon = ax0.axon_classified_dict[self.model]["all"]
        df_axon = df_axon.rename(columns={'prob': 'axon_prob'})
        _lst0 = df_axon["nrn"].tolist()


        if self.with_mix_point:
            mx0 = Classify_Mix(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.mx_feature_type, self.with_replacement)
            mx0 = mx0.classify_data()

            df_mix = mx0.mix_classified_dict[self.model]["all"]
            df_mix = df_mix.rename(columns={'prob': 'mix_prob'})
            _lst1 = df_mix["nrn"].tolist()
            _lst = list(set(_lst0) & set(_lst1))

        else:
            _lst = _lst0


        nrn_lst = list_unique(_lst)


        for nrn in nrn_lst:
            self.nrn_name = nrn

            ### fname
            self.tree_type = "result_tree"

            # _methodLevel
            if all([self.remove_method is None, self.target_level is None]):
                _fname0 = self.nrn_name

            elif all([type(self.remove_method) is str, type(self.target_level) is int]):
                # _methodLevel
                _methodLevel = remove_method + str(target_level)
                _fname0 = "_".join([self.nrn_name, _methodLevel])

            # _replaceTimes
            if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
                _replaceTimes = "rep" + str(self.with_replacement)
                _fname0 = "_".join([_fname0, _replaceTimes])

            # _trainTest
            _dict = {"all": "A", "normal": "N", "multiple": "M", "small": "S", "axonNear": "Nr"}
            if self.train_nrn_type == self.test_nrn_type:
                _trainTest = _dict[self.train_nrn_type[0]]
            else:
                _train = _dict[self.train_nrn_type[0]]
                _test = _dict[self.test_nrn_type[0]]
                _trainTest = "".join([_train, _test])
            _fname0 = "_".join([_fname0, _trainTest])

            # _featureType
            if not self.with_mix_point:
                _ax_mx_feature = self.ax_feature_type
            else:
                _ax_mx_feature = "".join([self.ax_feature_type, self.mx_feature_type])
            _fname0 = "_".join([_fname0, _ax_mx_feature])

            # # _withMixPoint
            # if self.with_mix_point:
            #     _fname0 = "_".join([_fname0, "mx"])

            self.save_path = input_folder + "nrn_plot/" + self.tree_type + "/" + self.model + "/"
            self.output_name = _fname0
            self.fname = self.save_path + self.output_name + ".gv." + self.file_type


            ### Read in data
            n1 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
            n1 = n1.load_data()
            df1 = n1.df
            df_dis1 = n1.df_dis
            polarity_dict = n1.polarity_dict
            order_lst_0 = df1['ID'].tolist()

            _df_axon = df_axon.loc[df_axon["nrn"]==nrn, [self.child_col, "axon_prob"]]
            if self.with_mix_point:
                _df_mix = df_mix.loc[df_mix["nrn"] == nrn, [self.child_col, "mix_prob"]]
                df_r = pd.merge(_df_axon, _df_mix, how="left", on=self.child_col)
            else:
                df_r = _df_axon

            df_r = df_r.rename(columns={'ID': 'descendant'})



            ### Plot result level tree
            if self.with_mix_point:
                neuron_plot_realPredMix_tree(df_dis1, df_r, 'descendant', 'ancestor', 'len', 'axon_prob', 'mix_prob',
                                             polarity_dict,
                                             node_order_lst=order_lst_0, save_path=self.save_path, subname=False,
                                             filename=self.output_name, file_type=self.file_type,
                                             fig_size=self.file_size)

            else:
                neuron_plot_prob_tree(df_dis1, df_r, 'descendant', 'ancestor', 'len', 'axon_prob',
                                      polarity_dict,
                                      node_order_lst=order_lst_0, save_path=self.save_path, subname=False,
                                      filename=self.output_name, file_type=self.file_type, fig_size=self.file_size)


        return


if __name__ == "__main__":
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for model in bar(model_lst):

        tree = Plot_Result_Tree(model, input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, mx_feature_type, with_replacement, with_mix_point, file_type)

        tree.plot_tree()

    time.sleep(0.01)
    
########################################################################################################################
# End of Code
########################################################################################################################
