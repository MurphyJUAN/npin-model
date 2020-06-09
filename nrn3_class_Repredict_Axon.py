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
from nrn3_class_Classify_Axon import Classify_Axon
from nrn3_class_Classify_Mix import Classify_Mix
from nrn3_class_Data_Cleaner import Data_Cleaner



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
ax_feature_type = 'af8'
with_replacement = 50
overwrite = False
seed=123


'''
"af1": n_s,
"af2": n_l,
"af3": n_ds,
"af4": n_s + n_l,
"af5": n_s + n_ds,
"af6": n_l + n_ds,
"af7": n_s + n_l + n_ds,
"af8": s + n_s + ds + n_ds + n_dtm,
"af9": s + n_s + ds + n_ds,
"af8-5": ["s_00000", "norm_s_00000", "norm_diff_trm_00000", "norm_direct_s_00000", "direct_s_00000"],
"af8-1": ["s_00000"],
"af9-5": ["s_00000", "norm_s_00000", "norm_direct_s_00000", "direct_s_00000", "s_10000"],
"af9-1": ["s_00000"],
"af10": s + n_s + ds + n_ds + l + n_l, 
"P1-1": P_ds,
"P1-2": P_nds,
"P1-3": P_s,
"P1-4": P_ns,
"P2-1": P2_dsnds,
"P2-2": P2_sns,
"P2-3": P2_dss,
"P2-4": P2_ndsns,
"P3-1": P3_snsnds
'''


########################################################################################################################
# Main Code
########################################################################################################################
class Repredict_Axon:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 ax_feature_type='af4',
                 sample_with_replacement=None,
                 seed=None,
                 pred_prob=None,
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                 ):

        self.repredict_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.ax_feature_type = ax_feature_type
        self.with_replacement = sample_with_replacement
        self.seed = seed
        self.pred_prob = pred_prob
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        self.branch = 2
        self.sampling_pct = 0.2
        # self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["repredict", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["repredict", self.classify_target, _methodLevel])

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

        self.fname = input_folder + "nrn_result/" + _fname3 + ".pkl"



        # Models
        ridge_class = RidgeClassifier()
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True, gamma="scale")
        xgb_class = XGBClassifier()
        # lgb_class = lgb.LGBMClassifier()
        # self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}
        # self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'xgb': xgb_class}
        self.models = {'rf': rf_class, 'xgb': xgb_class}
        # self.models = {'rf': rf_class, 'lgb': lgb_class}


        return




    def evaluation_info(self):
        if self._is_ready():
            self._load_data()
            for key, v0 in self.repredict_dict.items():
                if key == "info_dict":
                    print_info_dict(self.repredict_dict[key])
                    print("===========================================================================================")
                elif key != "sample_set":
                    print("Model =", key)
                    print("confusion_matrix:\n", turn_confusingMatrix_to_DF(self.repredict_dict[key]["confusion_matrix"]))
                    print("evaluation_df1:\n", self.repredict_dict[key]["evaluation_df1"], "\n")
                    print("evaluation_accuracy (by neuron):\n", self.repredict_dict[key]["evaluation_accuracy"], "\n")
                    print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return


    def repredict_axon(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._load_data_from_Classify_Axon_Mix()
            self._data_info()
            self._repredict()
            self._evaluate()
            self._save_result()
        return self


    def _is_ready(self):
        if os.path.exists(self.fname):
            return True
        else:
            return False


    def _load_data(self):
        with open(self.fname, "rb") as file:
            self.repredict_dict = pickle.load(file)
        return


    def _data_info(self):
        self.repredict_dict["info_dict"] = {
            "target": self.classify_target,
            "train_nrn_type": self.train_nrn_type,
            "test_nrn_type": self.test_nrn_type,
            "remove_method": self.remove_method,
            "target_level": self.target_level,
            "ax_feature_type": self.ax_feature_type,
            "sample_with_replacement_times": self.with_replacement
        }

        return


    def _load_data_from_Classify_Axon_Mix(self):
        ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement, seed=seed)
        # mx0 = Classify_Mix(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.mx_feature_type, self.with_replacement, self.seed)

        ax0 = ax0.classify_data()
        # mx0 = mx0.classify_data()

        self.ax0 = ax0.axon_classified_dict
        # self.mx0 = mx0.mix_classified_dict

        return


    def _repredict(self):
        print("Repredict Axon...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            model_dict = {}
            df_final = None
            df_a = self.ax0[model_name]["all"]
            # df_m = self.mx0[model_name]["all"]

            _lst0 = df_a["nrn"].unique().tolist()
            # _lst1 = df_m["nrn"].unique().tolist()
            # nrn_lst = sorted(list(set(_lst0) & set(_lst1)))
            nrn_lst = sorted(_lst0)


            for idx0, nrn in enumerate(nrn_lst):
                nrn0 = Data_Cleaner(input_folder, nrn)
                n0 = nrn0.load_data()
                # df_dis = n0.df_dis
                pld = n0.polarity_dict
                tnd = n0.tree_node_dict

                df = n0.df[[self.child_col, self.parent_col, self.type_col]].copy()
                df = add_bushCol(df, tnd, self.child_col, self.parent_col, descendant_only_forkLeaf=False)


                # m_pred = sorted(df_m.loc[(df_m["inference"] == 1) & (df_m["nrn"] == nrn), self.child_col].tolist(), reverse=True)
                # a_pred = df_a.loc[(df_a["prob"] >= 0.9) & (df_a["nrn"] == nrn), self.child_col].tolist()
                a_pred = df_a.loc[(df_a["prob"] > 0.5) & (df_a["nrn"] == nrn), self.child_col].tolist()

                repredict_dict = {"axon":[], "dendrite":[]}
                b0 = set(df.loc[df[self.child_col]==tnd["root"][0], "bush"].values[0])
                for a in a_pred:
                    ab0 = b0 & set(df.loc[df[self.child_col]==a, "bush"].values[0])
                    b0 = b0 - ab0

                    repredict_dict["axon"] += list(ab0)

                repredict_dict["dendrite"] += list(b0)



                a_leaf = list(set(tnd["leaf"])&set(pld["axon"]))
                d_leaf = list(set(tnd["leaf"])&set(pld["dendrite"]))
                na_leaf = list(set(tnd["leaf"])-set(a_leaf)-set(d_leaf))

                df_re = df.loc[df[self.child_col].isin(a_leaf + d_leaf)].copy()
                df_re["nrn"] = nrn
                df_re = df_re[["nrn", self.child_col, self.type_col]].sort_values([self.child_col]).reset_index(drop=True)
                df_re["label"] = np.where(df_re[self.child_col].isin(a_leaf), 1, 0)
                df_re["inference"] = np.where(df_re[self.child_col].isin(repredict_dict["axon"]), 1, 0)

                if df_final is None:
                    df_final = df_re
                else:
                    df_final = pd.concat([df_final, df_re])

            model_dict["df_repredict"] = df_final.reset_index(drop=True)

            self.repredict_dict[model_name] = model_dict




        time.sleep(0.01)

        return


    def _evaluate(self):
        print("Evaluate models...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for m in bar(list(self.models.keys())):
            df_eval = pd.DataFrame()

            acc_lst = []
            prc_lst = []
            rcl_lst = []
            f1_lst  = []

            # Calculate the result of Average of all tests (test_df3)
            df_re = self.repredict_dict[m]["df_repredict"]

            # 1. Evaluation DF and confusing matrix
            df_re1 = df_re.copy()
            y_test = df_re1['label'].values
            y_pred = df_re1['inference'].values

            acc_lst.append(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, output_dict=True)
            prc_lst.append(report["1"]["precision"])
            rcl_lst.append(report["1"]["recall"])
            f1_lst.append(report["1"]["f1-score"])

            cm = confusion_matrix(y_test, y_pred)

            df_eval["accuracy"] = acc_lst
            df_eval["precision"] = prc_lst
            df_eval["recall"] = rcl_lst
            df_eval["f1-score"] = f1_lst


            # 2. Accuracy by neuron
            nrn_lst = df_re.nrn.unique()
            c_lst = []
            w_lst = []
            for nrn in nrn_lst:
                _df = df_re.loc[df_re['nrn']==nrn].reset_index(drop=True)
                _t = _df["label"].tolist()
                _p = _df["inference"].tolist()
                if _t == _p:
                    c_lst.append(nrn)
                else:
                    w_lst.append(nrn)
            c_ratio = len(c_lst) / len(nrn_lst)

            self.repredict_dict[m]["confusion_matrix"] = cm
            self.repredict_dict[m]["evaluation_df1"] = df_eval
            self.repredict_dict[m]["evaluation_accuracy"] = c_ratio
            self.repredict_dict[m]["correct_pred"] = c_lst
            self.repredict_dict[m]["wrong_pred"] = w_lst


        time.sleep(0.01)

        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.repredict_dict, file=file)

        return





if __name__ == '__main__':
    # for t in ["med", "pcb"]:
    #     for i in ["P3-1"]:
    #     # for i in ["P1-1", "P1-2", "P1-3", "P1-4", "P2-1", "P2-2", "P2-3", "P2-4"]:
    #         axmx0 = Repredict_Axon(input_folder, [t], [t], remove_method, target_level, i, with_replacement, seed, overwrite=overwrite)
    #
    #         axmx0 = axmx0.repredict_axon()
    #
    #         axmx0.evaluation_info()
    #         print("===========================================================================================")

    # for i in ['af8']:
    #     axmx0 = Repredict_Axon(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, i, with_replacement, seed, overwrite=overwrite)
    #
    #     axmx0 = axmx0.repredict_axon()
    #
    #     axmx0.evaluation_info()
    #     print("===========================================================================================")

    axmx0 = Repredict_Axon(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, with_replacement, seed, overwrite=overwrite)

    axmx0 = axmx0.repredict_axon()

    axmx0.evaluation_info()




########################################################################################################################
