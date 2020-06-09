# Purpose:
# 1. Create feature for each node (DO NOT drop first fork):
# csu:
# len
# len_des_soma
# direct_dis_des_soma
#
#
# 2. label={'axon':1, 'dendrite':0}
# 3. Output:
#       prepared_axon_leaf5.pkl: prepare nrns.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Create_Axon_Feature import Create_Axon_Feature
from nrn3_class_Classify_Axon import Classify_Axon


########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
nrn_lst = neuron_dict["all"]
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
ax_feature_type = 'af8'
with_replacement = 50
# ax_train_with_only_axon_dendrite = True
overwrite = False


########################################################################################################################
# Main Code
########################################################################################################################
class Create_Mix_Feature:

    def __init__(self,
                 input_folder,
                 nrn_name,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 ax_feature_type="af4",
                 sample_with_replacement=None,
                 seed=123,
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID', 'NC', 'level', 'dp_level', 'branch', 'Q', 'L', 'L_sort', 'bush'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                 ):

        if all([remove_method is None, target_level is None]):
            self.fname = input_folder+"nrn_cleaned/"+nrn_name+".pkl"
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method+str(target_level)
            self.fname = input_folder+"nrn_cleaned/"+"_".join([nrn_name, _methodLevel])+".pkl"
        else:
            sys.exit("\n remove_method = None or str; target_level = None or int! Check Prepare_Axon.")

        self.input_folder = input_folder
        self.nrn_name = nrn_name
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.ax_feature_type = ax_feature_type
        self.with_replacement = sample_with_replacement
        # self.ax_train_only_axdn = ax_train_with_only_axon_dendrite
        self.seed = seed
        self.origin_cols = origin_cols
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        ### Add new features
        # _trainTest
        _dict = {"all": "A", "normal": "N", "multiple": "M", "small": "S", "axonNear": "Nr"}
        if self.train_nrn_type == self.test_nrn_type:
            _trainTest = _dict[self.train_nrn_type[0]]
        else:
            _train = _dict[self.train_nrn_type[0]]
            _test = _dict[self.test_nrn_type[0]]
            _trainTest = "".join([_train, _test])

        # _replaceTimes
        if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
            _replaceTimes = "rep" + str(self.with_replacement)
            _fname0 = "_".join([_replaceTimes, _trainTest])
        else:
            _fname0 = _trainTest

        # _ax_feature
        _fname1 = "_".join([_fname0, self.ax_feature_type])

        self.feature_info = {}
        for m in ["gbdt", "rf", "svm", "xgb"]:
            _fname2 = "_".join([m, _fname1])
            self.feature_info[_fname2] = m


        self.branch = 2


        return


    def load_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
            if self._is_feature_ready():
                pass
            else:
                self._create_df_from_Axon()
                self._save_data()
                self._load_data()
        else:
            self._create_df_from_Axon()
            self._save_data()
            self._load_data()
        return self


    def _is_ready(self):
        if os.path.exists(self.fname):
            return True
        else:
            return False


    def _load_data(self):
        with open(self.fname, "rb") as file:
            self.df, self.df_dis, self.df_axon, self.df_mix, self.df_level_descend, self.tree_node_dict, self.polarity_dict, self.axon_feature, self.mix_feature = pickle.load(file)
        return


    def _is_feature_ready(self):
        if self.df_mix is None:
            return False
        else:
            col_lst = list(self.df_mix)
            ftr_lst = list(self.feature_info.keys())
            if all(elem in col_lst for elem in ftr_lst):
                return True
            else:
                return False


    def _create_df_from_Axon(self):
        ### Read in data
        n1 = Create_Axon_Feature(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
        n1 = n1.load_data()
        df_mix = n1.df_mix
        mix_lst = n1.polarity_dict["mix"]

        ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement, seed=self.seed)
        ax0 = ax0.classify_data()

        ### 1. Inherit feature (length/distance) from axon_df
        if any([df_mix is None, self.overwrite]):
            df = n1.df_axon.drop(['label'], 1)
            axon_feature = n1.axon_feature

            ### Update ftr info
            feature_info = merge_two_dicts(axon_feature, self.feature_info)

        else:
            df = n1.df_mix.drop(['label'], 1)
            mix_feature = n1.mix_feature

            ### Update ftr info
            feature_info = merge_two_dicts(mix_feature, self.feature_info)


        ### 2. Add feature (prob)
        for k, v in self.feature_info.items():
            df_ftr = ax0.axon_classified_dict[v]["all"]
            df_ftr = df_ftr.loc[df_ftr["nrn"] == self.nrn_name, [self.child_col, "prob"]]
            df_ftr = df_ftr.rename(columns={"prob": k})
            df = pd.merge(df, df_ftr, how="left", on=self.child_col)
            df[k] = np.where(df[self.child_col] == 1, 0, df[k])


        ### Add label
        df['label'] = np.where(df[self.child_col].isin(mix_lst), 1, 0)


        ### Update data
        self.n1 = n1
        self.n1.df_mix = df
        self.n1.mix_feature = feature_info

        return


    def _save_data(self):
        with open(self.fname, "wb") as file:
                pickle.dump([self.n1.df, self.n1.df_dis, self.n1.df_axon, self.n1.df_mix, self.n1.df_level_descend, self.n1.tree_node_dict, self.n1.polarity_dict, self.n1.axon_feature, self.n1.mix_feature], file=file)

        return


if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for nrn in bar(nrn_lst):
        mx0 = Create_Mix_Feature(input_folder, nrn, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, with_replacement, overwrite=overwrite)
        mx0 = mx0.load_data()

    time.sleep(0.01)


########################################################################################################################
# End of Code
########################################################################################################################
