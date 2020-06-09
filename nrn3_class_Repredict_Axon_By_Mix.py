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
remove_method = "leaf"
target_level = 5
ax_feature_type = 'af9'
mx_feature_type = 'mf10'
with_replacement = 50
overwrite = False
seed=123

'''
"mf1": n_s,
"mf2": n_l,
"mf3": n_ds,
"mf4": n_s + n_l,
"mf5": n_s + n_ds,
"mf6": n_l + n_ds,
"mf7": n_s + n_l + n_ds,
"mf8": n_s + n_l + pr,
"mf9": n_s + n_l + npr,
"mf10": n_s + n_l + ppr,
"mf11": pr,
"mf12": npr,
"mf13": ppr,
"mf14": s + n_s + ds + n_ds + n_dtm,
"mf15": s + n_s + ds + n_ds,
"mf16": s + n_s + ds + n_ds + pr,
"mf17": s + n_s + ds + n_ds + npr,
"mf18": s + n_s + ds + n_ds + ppr
'''


########################################################################################################################
# Main Code
########################################################################################################################
class Repredict_Axon_By_Mix:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 ax_feature_type='af4',
                 mx_feature_type='mf4',
                 sample_with_replacement=None,
                 seed=None,
                 pred_prob=None,
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                 ):

        self.repredict_dict = {}
        self.classify_target = "axonMix"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.ax_feature_type = ax_feature_type
        self.mx_feature_type = mx_feature_type
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
        self.base_prob = 0.5
        self.high_prob = 0.95

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
        _ax_mx_feature = "".join([self.ax_feature_type, self.mx_feature_type])
        _fname3 = "_".join([_fname2, _ax_mx_feature])

        self.fname = input_folder + "nrn_result/" + _fname3 + ".pkl"



        # Models
        ridge_class = RidgeClassifier()
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True, gamma="scale")
        xgb_class = XGBClassifier()
        self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}


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
                    for i in ["level", "reduced"]:
                        if i == "level":
                            name0 = i + " tree (terminal):"
                        else:
                            name0 = i + " tree (all nodes):"
                        print(name0, "\n")
                        print("confusion_matrix:\n", self.repredict_dict[key]["confusion_matrix_" + i])
                        print("confusion_matrix_highProb:\n", self.repredict_dict[key]["confusion_matrix_highProb_" + i])
                        print("evaluation_df1:\n", self.repredict_dict[key]["evaluation_df1_" + i], "\n")
                        # print("evaluation_accuracy (by neuron):\n", self.repredict_dict[key]["evaluation_accuracy"], "\n")
                        print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return


    def repredict_axon_by_mix(self):
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
            "mx_feature_type": self.mx_feature_type,
            "sample_with_replacement_times": self.with_replacement
        }

        return


    def _load_data_from_Classify_Axon_Mix(self):
        ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement, seed=seed)
        mx0 = Classify_Mix(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.mx_feature_type, self.with_replacement, self.seed)

        ax0 = ax0.classify_data()
        mx0 = mx0.classify_data()

        self.ax0 = ax0.axon_classified_dict
        self.mx0 = mx0.mix_classified_dict

        return


    def _repredict(self):
        print("Repredict Axon by Mix...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            model_dict = {}
            df_final0 = None
            df_final1 = None
            df_a = self.ax0[model_name]["all"]
            df_m = self.mx0[model_name]["all"]

            _lst0 = df_a["nrn"].unique().tolist()
            _lst1 = df_m["nrn"].unique().tolist()
            nrn_lst = sorted(list(set(_lst0) & set(_lst1)))


            for idx0, nrn in enumerate(nrn_lst):
                # Read in original (level) nrn
                nrn0 = Data_Cleaner(self.input_folder, nrn)
                n0 = nrn0.load_data()
                df_dis0 = n0.df_dis
                pld_level = n0.polarity_dict
                tnd_level = n0.tree_node_dict

                df0 = n0.df[[self.child_col, self.parent_col, self.type_col]].copy()
                df0 = add_bushCol(df0, tnd_level, self.child_col, self.parent_col, descendant_only_forkLeaf=False)

                # Read in reduced nrn
                nrn1 = Data_Cleaner(self.input_folder, nrn, self.remove_method, self.target_level)
                n1 = nrn1.load_data()
                pld_reduced = n1.polarity_dict
                tnd_reduced = n1.tree_node_dict


                m_pred = sorted(df_m.loc[(df_m["inference"] == 1) & (df_m["nrn"] == nrn), self.child_col].tolist(), reverse=True)
                a_pred0 = df_a.loc[(df_a["prob"] >  self.base_prob) & (df_a["nrn"] == nrn), self.child_col].tolist()
                a_pred1 = df_a.loc[(df_a["prob"] >= self.high_prob) & (df_a["nrn"] == nrn), self.child_col].tolist()


                repredict_dict = {"axon0":[], "dendrite0":[], "axon1":[], "dendrite1":[]}
                repredict_dict["axon0"] += a_pred0
                repredict_dict["axon1"] += a_pred1

                b0 = set(df0.loc[df0[self.child_col]==tnd_level["root"][0], "bush"].values[0])
                for m in m_pred:
                    mb0 = b0 & set(df0.loc[df0[self.child_col]==m, "bush"].values[0])
                    b0 = b0 - mb0

                    d_lst = df_dis0.loc[df_dis0["ancestor"]==m, "descendant"].tolist()
                    for d in d_lst:
                        ab0 = mb0 & set(df0.loc[df0[self.child_col] == d, "bush"].values[0])

                        # 1. Find prob of axon > 0.5
                        if ab0 & set(a_pred0):
                            repredict_dict["axon0"] += list(ab0)
                        else:
                            repredict_dict["dendrite0"] += list(ab0)

                        # 2. Find prob of axon >= 0.95
                        if ab0 & set(a_pred1):
                            repredict_dict["axon1"] += list(ab0)
                        else:
                            repredict_dict["dendrite1"] += list(ab0)

                # 1. Find prob of axon > 0.5
                if b0 & set(a_pred0):
                    repredict_dict["axon0"] += list(b0)
                else:
                    repredict_dict["dendrite0"] += list(b0)

                # 2. Find prob of axon >= 0.95
                if b0 & set(a_pred1):
                    repredict_dict["axon1"] += list(b0)
                else:
                    repredict_dict["dendrite1"] += list(b0)


                # 1. The terminal accuracy of level tree
                a_leaf = list(set(tnd_level["leaf"])&set(pld_level["axon"]))
                # d_leaf = list(set(tnd_level["leaf"])&set(pld_level["dendrite"]))
                # na_leaf = list(set(tnd_level["leaf"])-set(a_leaf)-set(d_leaf))

                # df_level = df0.loc[df0[self.child_col].isin(a_leaf + d_leaf)].copy()
                df_level = df0.loc[df0[self.child_col].isin(tnd_level["leaf"])].copy()
                df_level["nrn"] = nrn
                df_level = df_level[["nrn", self.child_col, self.type_col]].sort_values([self.child_col]).reset_index(drop=True)
                df_level["label"] = np.where(df_level[self.child_col].isin(a_leaf), 1, 0)
                df_level["inference_" + str(self.base_prob)] = np.where(df_level[self.child_col].isin(repredict_dict["axon0"]), 1, 0)
                df_level["inference_" + str(self.high_prob)] = np.where(df_level[self.child_col].isin(repredict_dict["axon1"]), 1, 0)

                if df_final0 is None:
                    df_final0 = df_level
                else:
                    df_final0 = pd.concat([df_final0, df_level])


                # 2. The all node accuracy of reduced tree
                tn_lst = dict_merge_value(tnd_reduced)
                df_reduced = df0.loc[df0[self.child_col].isin(tn_lst)].copy()
                df_reduced["nrn"] = nrn
                df_reduced = df_reduced[["nrn", self.child_col, self.type_col]].sort_values([self.child_col]).reset_index(drop=True)
                df_reduced["label"] = np.where(df_reduced[self.child_col].isin(pld_reduced["axon"]), 1, 0)
                df_reduced["inference_" + str(self.base_prob)] = np.where(df_reduced[self.child_col].isin(repredict_dict["axon0"]), 1, 0)
                df_reduced["inference_" + str(self.high_prob)] = np.where(df_reduced[self.child_col].isin(repredict_dict["axon1"]), 1, 0)

                if df_final1 is None:
                    df_final1 = df_reduced
                else:
                    df_final1 = pd.concat([df_final1, df_reduced])


            model_dict["df_level"] = df_final0.reset_index(drop=True)
            model_dict["df_reduced"] = df_final1.reset_index(drop=True)

            self.repredict_dict[model_name] = model_dict




        time.sleep(0.01)

        return


    def _evaluate(self):
        print("Evaluate models...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for m in bar(list(self.models.keys())):

            # Calculate the result of level tree
            for i in ["level", "reduced"]:
                df_re0 = self.repredict_dict[m]["df_" + i]

                acc_lst = []
                prc_lst = []
                rcl_lst = []
                f1_lst = []
                cm_lst = []

                for prob in [self.base_prob, self.high_prob]:
                    # 1. Evaluation DF and confusing matrix
                    df_re1 = df_re0.copy()
                    y_test = df_re1['label'].values
                    y_pred = df_re1['inference_' + str(prob)].values

                    acc_lst.append(accuracy_score(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    prc_lst.append(report["1"]["precision"])
                    rcl_lst.append(report["1"]["recall"])
                    f1_lst.append(report["1"]["f1-score"])

                    cm = confusion_matrix(y_test, y_pred)
                    cm_lst.append(cm)


                    # # 2. Accuracy by neuron
                    # nrn_lst = df_re0.nrn.unique()
                    # c_lst = []
                    # w_lst = []
                    # for nrn in nrn_lst:
                    #     _df = df_re0.loc[df_re0['nrn']==nrn].reset_index(drop=True)
                    #     _t = _df["label"].tolist()
                    #     _p = _df["inference"].tolist()
                    #     if _t == _p:
                    #         c_lst.append(nrn)
                    #     else:
                    #         w_lst.append(nrn)
                    # c_ratio = len(c_lst) / len(nrn_lst)


                df_eval = pd.DataFrame({"accuracy": acc_lst,
                                        "precision": prc_lst,
                                        "recall": rcl_lst,
                                        "f1-score": f1_lst},
                                       index = [self.base_prob, self.high_prob])

                df_eval = df_eval[["accuracy", "precision", "recall", "f1-score"]]

                self.repredict_dict[m]["confusion_matrix_" + i] = cm_lst[0]
                self.repredict_dict[m]["confusion_matrix_highProb_" + i] = cm_lst[1]
                self.repredict_dict[m]["evaluation_df1_" + i] = df_eval
                # self.repredict_dict[m]["evaluation_accuracy"] = c_ratio
                # self.repredict_dict[m]["correct_pred"] = c_lst
                # self.repredict_dict[m]["wrong_pred"] = w_lst


        time.sleep(0.01)

        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.repredict_dict, file=file)

        return





if __name__ == '__main__':
    # for i in ['af8']:
    #     for j in ['mf4', 'mf8', 'mf9', 'mf10', 'mf11', 'mf14']:
    #         axmx0 = Repredict_Axon_By_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, i, j, with_replacement, seed, overwrite=overwrite)
    #
    #         axmx0 = axmx0.repredict_axon_by_mix()
    #
    #         axmx0.evaluation_info()
    #         print("===========================================================================================")

    axmx0 = Repredict_Axon_By_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, mx_feature_type, with_replacement, seed, overwrite=overwrite)

    axmx0 = axmx0.repredict_axon_by_mix()

    axmx0.evaluation_info()




########################################################################################################################
