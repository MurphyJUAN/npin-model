# Purpose:
# 1. Data point: each node in level tree WITHOUT soma
# 2. features=[s0, d0, d1,...] by using only reduced level tree & DO NOT drop first fork
# 3. label={'axon':1, 'dendrite':0}
# 4. Create df with features and label
# 5. Output:
#       prepared_axon_leaf5.pkl: prepare nrns.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Create_Mix_Feature import Create_Mix_Feature


########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
ax_feature_type = 'af8'
with_replacement = 50
overwrite = False



########################################################################################################################
# Main Code
########################################################################################################################
class Prepare_Mix:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 ax_feature_type='af4',
                 sample_with_replacement=None,
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                 ):

        self.input_folder = input_folder
        self.nrn_lst = neuron_dict["all"]
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.ax_feature_type = ax_feature_type
        self.with_replacement = sample_with_replacement
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["prepared", "mix"])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["prepared", "mix", _methodLevel])

        # _replaceTimes
        if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
            _replaceTimes = "replace" + str(self.with_replacement)
            _fname1 = "_".join([_fname0, _replaceTimes])
        else:
            _fname1 = _fname0

        # _trainTest
        if self.train_nrn_type == self.test_nrn_type:
            _trainTest = '&'.join(self.train_nrn_type)
            _fname2 = "_".join([_fname1, _trainTest])
        else:
            _train = '&'.join(self.train_nrn_type)
            _test = '&'.join(self.test_nrn_type)
            _fname2 = "_".join([_fname1, _train, _test])

        # _featureType
        _fname3 = "_".join([_fname2, self.ax_feature_type])

        self.fname = input_folder + "nrn_cleaned/" + _fname3 + ".pkl"


        # Select mix features
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
        else:
            self._create_df_from_Create_Mix_Feature()
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
            self.mix_prepared_df = pickle.load(file)
        return


    def _create_df_from_Create_Mix_Feature(self):
        print("Prepare Mix...")
        df_final = None
        feature_level = 5
        self.L_lst, self.L_sort_lst, _ = create_total_L(feature_level, self.branch)

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for nrn in bar(self.nrn_lst):
            b1 = bar.value

            ### Read in reduced df from Data_cleaner
            n1 = Create_Mix_Feature(self.input_folder, nrn, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement)
            n1 = n1.load_data()
            df = n1.df_mix
            axon_ftr = n1.axon_feature
            feature_info = merge_two_dicts(axon_ftr, self.feature_info)
            tnd = n1.tree_node_dict


            ### Find first fork
            # _, orig_ff = assign_root_as_firstFork(df_dis1, tnd, child_col='descendant', parent_col='ancestor')

            ### Select data points (every node except soma)
            data_points = list(set(tnd['fork']) & set(df[self.child_col]))
            # data_points = list((set(tnd['leaf']) | set(tnd['fork'])) & set(df[self.child_col]))
            # data_points = [x for x in data_points if x != orig_ff]


            ### Create feature & label
            # Create empty df_ftr
            df_ftr = pd.DataFrame({'cols': ['nrn', self.child_col, self.type_col, "label"]})
            for k, v in feature_info.items():
                _lst = [v + "_" + i for i in self.L_sort_lst]
                if v in ["gbdt", "rf", "svm", "xgb"]:
                    _df = pd.DataFrame({'cols': _lst + [v + "_npr_0"] + [v + "_npr_1"] + [v + "_ppr_0"] + [v + "_ppr_1"]})
                else:
                    _df = pd.DataFrame({'cols': _lst})
                df_ftr = pd.concat([df_ftr, _df])
            df_ftr = df_ftr.reset_index(drop=True)

            # Run through data points
            for idx2, d in enumerate(data_points):
                b2 = idx2
                b_value = b1 * (10 ** (num_digits(len(self.L_sort_lst)))) + b2

                # Find the type & label of the data point
                t = df.loc[df[self.child_col] == d, self.type_col].values[0]
                l = df.loc[df[self.child_col] == d, 'label'].values[0]

                # Create sub bush with new L for the data point
                df_sub = new_L_for_sub_bush(df, self.child_col, d, feature_level)

                df_id = pd.DataFrame({'cols': ['nrn', self.child_col, self.type_col, 'label'], b_value: [nrn, d, t, l]})

                df_d = None
                for k, v in feature_info.items():
                    # 1. Create features: length/distance, prob
                    _df = pd.DataFrame({'cols': v + "_" + df_sub['L_sort'], b_value: df_sub[k]})

                    # 2. Create features: npr, ppr
                    if v in ["gbdt", "rf", "svm", "xgb"]:

                        df_left = df_sub[df_sub['L_sort'].str.startswith('1')]
                        df_right = df_sub[df_sub['L_sort'].str.startswith('2')]

                        # create npr_i
                        df_npr_0 = df_left.loc[df_left[k] > 0.5].copy()
                        df_npr_1 = df_right.loc[df_right[k] > 0.5].copy()
                        npr_0, npr_1 = len(df_npr_0), len(df_npr_1)
                        # Pass the row elements as key value pairs to append() function
                        _df = _df.append({'cols': v + "_npr_0", b_value: npr_0}, ignore_index=True)
                        _df = _df.append({'cols': v + "_npr_1", b_value: npr_1}, ignore_index=True)


                        # create ppr_i
                        if len(df_left) != 0:
                            ppr_0 = npr_0 / len(df_left)
                        else:
                            ppr_0 = 0
                        if len(df_right) != 0:
                            ppr_1 = npr_1 / len(df_right)
                        else:
                            ppr_1 = 0
                        _df = _df.append({'cols': v + "_ppr_0", b_value: ppr_0}, ignore_index=True)
                        _df = _df.append({'cols': v + "_ppr_1", b_value: ppr_1}, ignore_index=True)


                    if df_d is None:
                        df_d = _df
                    else:
                        df_d = pd.concat([df_d, _df])

                df_temp = pd.concat([df_id, df_d])
                df_ftr = pd.merge(df_ftr, df_temp, how='left', on='cols')

            # Transpose, rename cols, and fillna(0) (df_ftr)
            df_ftr = df_ftr.T.reset_index(drop=True)
            df_ftr.columns = df_ftr.iloc[0]
            df_ftr = df_ftr[1:].reset_index(drop=True)
            df_ftr = df_ftr.fillna(0)

            # Merge df_ftr to df_final
            if df_final is None:
                df_final = df_ftr
            else:
                df_final = pd.concat([df_final, df_ftr])


        time.sleep(0.01)

        self.mix_prepared_df = df_final.reset_index(drop=True)

        return


    def _save_data(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.mix_prepared_df, file=file)
        return


if __name__ == '__main__':
    mx0 = Prepare_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, with_replacement, overwrite=overwrite)

    mx0 = mx0.load_data()

    df = mx0.mix_prepared_df
    df = df[["nrn", "ID"]]

    # _method_level = remove_method+str(target_level)
    # a0.axon_prepared_df.to_csv(input_folder + "_".join(["axon", _method_level])+'.csv', index=False)
    # df.to_csv(input_folder + _method_level + '.csv', index=False)


########################################################################################################################
# End of Code
########################################################################################################################
