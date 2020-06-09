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
from nrn3_class_Data_Cleaner import Data_Cleaner
from nrn3_class_Classify_Axon import Classify_Axon
from nrn3_class_Classify_Mix import Classify_Mix



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["normal"]  # "all", "normal", or "multiple"
test_nrn_type = ["normal"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
feature_list = ['f4', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
with_replacement = 50
overwrite = False
seed=123


########################################################################################################################
# Main Code
########################################################################################################################
class Compare_Table:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 feature_list=['f4'],
                 sample_with_replacement=None,
                 seed=None,
                 pred_prob=None,
                 child_col='ID',
                 overwrite=False
                 ):

        self.compare_table_dict = {}
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.with_replacement = sample_with_replacement
        self.seed = seed
        self.pred_prob = pred_prob
        self.child_col = child_col
        self.overwrite = overwrite

        self.branch = 2
        self.sampling_pct = 0.2
        self.feature_list = sorted(feature_list)
        # self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)
        self.pred_prob = 0.9


        # train & test nrn lst
        self.train_nrn_lst = []
        self.test_nrn_lst = []
        if self.train_nrn_type == self.test_nrn_type:
            for t in self.train_nrn_type:
                self.train_nrn_lst += neuron_dict[t]
            self.test_nrn_lst = self.train_nrn_lst
        else:
            for t in self.train_nrn_type:
                self.train_nrn_lst += neuron_dict[t]
            for t in self.test_nrn_type:
                self.test_nrn_lst += neuron_dict[t]


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "compareTable"
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["compareTable", _methodLevel])

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
        _featureType = "-".join(self.feature_list)
        _fname3 = "_".join([_fname2, _featureType])

        self.fname = input_folder + "nrn_result/" + _fname3 + ".pkl"





        # Models
        ridge_class = RidgeClassifier()
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True, gamma="scale")
        xgb_class = XGBClassifier()
        self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}


        return




    def table_info(self):
        if self._is_ready():
            self._load_data()
            for key, v0 in self.compare_table_dict.items():
                if key == "info_dict":
                    print_info_dict(self.compare_table_dict[key])
                    print("===========================================================================================")
                else:
                    print("Model =", key)
                    print("compare_table:\n", self.compare_table_dict[key])
                    print("===========================================================================================")


        else:
            self._create_compare_table()

        return


    def compare_table(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._create_compare_table()
            self._save_result()
        return self


    def _is_ready(self):
        if os.path.exists(self.fname):
            return True
        else:
            return False


    def _load_data(self):
        with open(self.fname, "rb") as file:
            self.compare_table_dict = pickle.load(file)
        return


    def _create_compare_table(self):
        print("Create Predict Table...")
        df_both_ct = pd.DataFrame(index=list(self.models.keys()))
        df_axon_ct = pd.DataFrame(index=list(self.models.keys()), columns=["ftr", "ftr_pct"])
        df_mix_ct = pd.DataFrame(index=list(self.models.keys()))

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for ftr in bar(self.feature_list):
            df_both_ct[ftr] = np.nan
            df_both_ct[ftr + "_pct"] = np.nan
            df_mix_ct[ftr] = np.nan
            df_mix_ct[ftr + "_pct"] = np.nan

            # Read in results
            ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.with_replacement)
            ax0 = ax0.classify_data()
            mx0 = Classify_Mix(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, ftr, self.with_replacement)
            mx0 = mx0.classify_data()
            self.compare_table_dict["info_dict"] = mx0.mix_classified_dict["info_dict"]

            for i in list(self.models.keys()):
                df_axon = ax0.axon_classified_dict[i]["all"]
                df_mix = mx0.mix_classified_dict[i]["all"]

                nrn_lst = df_mix["nrn"].tolist()
                nrn_lst = list_unique(nrn_lst)

                # Create blank dict & df
                dict = {'both_T': [], 'only_axon_T': [], 'only_mix_T': [], 'both_F': []}
                df_result1 = pd.DataFrame(index=list(dict.keys()), columns=['num', 'nrn_lst'])
                for nrn in nrn_lst:
                    # Read in Data Cleaner
                    n0 = Data_Cleaner(self.input_folder, nrn, self.remove_method, self.target_level)
                    n0 = n0.load_data()

                    df_axon1 = df_axon.loc[df_axon['nrn'] == nrn].copy()
                    df_mix1 = df_mix.loc[df_mix['nrn'] == nrn]

                    # Check mix
                    df_mix1 = df_mix1.sort_values(['prob', self.child_col], ascending=[False, False])  # child_col ascending=F: choose larger ID as mix point
                    real_mix = n0.polarity_dict["mix"]
                    pred_mix = np.array(df_mix1[self.child_col])[:len(real_mix)]
                    if set(real_mix) == set(pred_mix):
                        m = 1
                    else:
                        m = 0


                    # Check axon
                    pred_axon = df_axon1.loc[df_axon1["prob"] > self.pred_prob, self.child_col].tolist()
                    real_axon = n0.polarity_dict["axon"]
                    if set(pred_axon).issubset(set(real_axon)):
                        a = 1
                    else:
                        a = 0


                    if all([a == 1, m == 1]):
                        dict['both_T'].append(nrn)
                    elif all([a == 1, m == 0]):
                        dict['only_axon_T'].append(nrn)
                    elif all([a == 0, m == 1]):
                        dict['only_mix_T'].append(nrn)
                    else:
                        dict['both_F'].append(nrn)


                both_T = len(dict["both_T"])
                df_both_ct.ix[i, ftr] = both_T
                df_both_ct.ix[i, ftr + "_pct"] = both_T/len(nrn_lst)

                axon_T = len(dict["both_T"]) + len(dict["only_axon_T"])
                df_axon_ct.ix[i, "ftr"] = axon_T
                df_axon_ct.ix[i, "ftr_pct"] = axon_T / len(nrn_lst)

                mix_T = len(dict["both_T"]) + len(dict["only_mix_T"])
                df_mix_ct.ix[i, ftr] = mix_T
                df_mix_ct.ix[i, ftr + "_pct"] = mix_T / len(nrn_lst)

        time.sleep(0.01)


        self.compare_table_dict["both_table"] = df_both_ct
        self.compare_table_dict["axon_table"] = df_axon_ct
        self.compare_table_dict["mix_table"] = df_mix_ct



        return



    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.compare_table_dict, file=file)

        return





if __name__ == '__main__':
    pt0 = Compare_Table(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, feature_list, with_replacement, seed, overwrite=overwrite)

    pt0 = pt0.compare_table()

    pt0.table_info()
    print("===========================================================================================")

    # mx_dict = mx0.compare_table_dict
    # print(ax_dict["sample_set"])

    # mx0.evaluation_info()




########################################################################################################################
