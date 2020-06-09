# Purpose:
# 1. Create train & test set
# 2. Run model & Evaluate model
# 3. Outputs:
#       result_axon_leaf5_normal.pkl: train=test=normal, sampling the test & train set w/o replacement.
#       result_axon_leaf5_replace20_normal.pkl: train=test=normal, sampling w/ replacement for 20 times.
#       result_axon_leaf5_normal&small.pkl: train=normal, test=small, no sampling.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Prepare_Axon import Prepare_Axon
from nrn3_class_Classify_Axon import Classify_Axon
from nrn3_class_Data_Cleaner import Data_Cleaner



########################################################################################################################
# Set up
########################################################################################################################
input_folder = './data/'
train_nrn_type = ["all"]  # "all", "normal", "small", "multiple", or "axonNear"
test_nrn_type = ["all"]  # "all", "normal", "small", "multiple", or "axonNear"
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
feature_type='af8'
with_replacement = 50
augment_nrn_lst = neuron_dict["oddLevelTree"]
augment_number = 10
train_with_only_axon_dendrite = True
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
"P2-4": P2_ndsns
'''


########################################################################################################################
# Main Code
########################################################################################################################
class Find_Best_Axon_Feature_Combination:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 feature_type='af4',
                 sample_with_replacement=None,
                 augment_nrn_lst=None,
                 augment_number=0,
                 train_with_only_axon_dendrite=False,
                 seed=123,
                 pred_prob=None,
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                 ):

        self.axon_classified_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.with_replacement = sample_with_replacement
        self.augment_nrn_lst = augment_nrn_lst
        self.augment_number = augment_number
        self.train_only_axdn = train_with_only_axon_dendrite
        self.seed = seed
        self.pred_prob = pred_prob
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        # self.branch = 2
        # self.sampling_pct = 1/7
        self.feature_type = feature_type
        # self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)


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


        # # Exclude fake data from test_nrn_lst & Update sampling_pct
        # self.fk_nrn_name_lst = []
        # if all([isinstance(augment_nrn_lst, list), augment_number > 0]):
        #     # fake data name
        #     for nrn_name in augment_nrn_lst:
        #         self.fk_nrn_name_lst += ["fk" + str(i) + "_" + nrn_name for i in range(augment_number)]
        #
        #     # Update sampling_pct
        #     self.sampling_pct = (self.sampling_pct*len(self.test_nrn_lst))/(len(self.test_nrn_lst)-len(self.fk_nrn_name_lst))
        #
        # self.test_nrn_lst = list(set(self.test_nrn_lst) - set(self.fk_nrn_name_lst))



        # # test samples
        # if all([self.train_nrn_type == self.test_nrn_type, self.with_replacement is None]):
        #     self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct)
        #
        # elif all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
        #     target_num = len(self.train_nrn_lst)
        #     flat_num = 0
        #     while target_num != flat_num:
        #         self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct, with_replacement=True, sampling_times=self.with_replacement)
        #         flat_lst = [item for sublist in self.sample_set for item in sublist]
        #         flat_lst = list_unique(flat_lst)
        #         flat_num = len(flat_lst)
        #
        # elif self.train_nrn_type != self.test_nrn_type:
        #     self.sample_set = [self.test_nrn_lst]
        #     self.with_replacement = None
        #
        # self.axon_classified_dict["sample_set"] = self.sample_set


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["result", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["result", self.classify_target, _methodLevel])

        # _augment
        if any([augment_nrn_lst is None, augment_number == 0]):
            pass
        elif all([isinstance(augment_nrn_lst, list), type(augment_number) is int]):
            _augment = "fk" + str(augment_number)
            _fname0 = "_".join([_fname0, _augment])
        else:
            sys.exit("\n augment_nrn_lst = None or list; augment_number = int! Check Prepare_Axon.")

        # _replaceTimes
        if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
            _replaceTimes = "replace" + str(self.with_replacement)
            _fname0 = "_".join([_fname0, _replaceTimes])

        # _trainTest
        if self.train_nrn_type == self.test_nrn_type:
            _trainTest = '&'.join(self.train_nrn_type)
            _fname0 = "_".join([_fname0, _trainTest])
        else:
            _train = '&'.join(self.train_nrn_type)
            _test = '&'.join(self.test_nrn_type)
            _fname0 = "_".join([_fname0, _train, _test])

        # _train_with_only_axon_dendrite
        if self.train_only_axdn:
            _fname0 = "_".join([_fname0, "_trnAD"])


        # _featureType
        _fname0 = "_".join([_fname0, self.feature_type, "comb"])

        self.fname = input_folder + "nrn_result/" + _fname0 + ".pkl"


        # Models
        ridge_class = RidgeClassifier()
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True, gamma="scale")
        xgb_class = XGBClassifier()
        # lgb_class = lgb.LGBMClassifier()
        # self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}
        # self.models = {'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}
        # self.models = {'rf': rf_class, 'xgb': xgb_class}
        # self.models = {'rf': rf_class, 'lgb': lgb_class}
        self.models = {'xgb': xgb_class}
        # self.models = {'rf': rf_class}


        return




    def evaluation_info(self):
        if self._is_ready():
            self._load_data()
            for key, v0 in self.axon_classified_dict.items():
                if key == "info_dict":
                    print_info_dict(self.axon_classified_dict[key])
                    print("===========================================================================================")
                elif key != "sample_set":
                    print("Model =", key)
                    # print("confusion_matrix (high precision):\n", turn_confusingMatrix_to_DF(self.axon_classified_dict[key]["confusion_matrix_highPrc"]))
                    print("evaluation_df1:\n", self.axon_classified_dict[key]["evaluation_df1"], "\n")
                    print("evaluation_df1_all:\n", self.axon_classified_dict[key]["evaluation_df1_all"], "\n")
                    print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return


    def classify_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._load_data_from_Prepare_Axon()
            # self._data_info()
            self._run_model()
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
            self.axon_classified_dict = pickle.load(file)
        return


    # def _data_info(self):
    #     self.axon_classified_dict["info_dict"] = {
    #         "target": self.classify_target,
    #         "train_nrn_type": self.train_nrn_type,
    #         "test_nrn_type": self.test_nrn_type,
    #         "remove_method": self.remove_method,
    #         "target_level": self.target_level,
    #         "feature_type": self.feature_type,
    #         "data_point (train)": self.data_point,
    #         "ratio_of_label1 (train)": self.ration_of_label1,
    #         "sample_with_replacement_times": self.with_replacement,
    #         "train_with_only_axon_dendrite": self.train_only_axdn
    #     }
    #
    #     return


    def _load_data_from_Prepare_Axon(self):
        ax0 = Prepare_Axon(self.input_folder, self.remove_method, self.target_level, self.augment_nrn_lst, self.augment_number)
        ax0 = ax0.load_data()
        axon_prepared_df = ax0.axon_prepared_df
        if self.train_only_axdn:
            self.train = axon_prepared_df.loc[(axon_prepared_df["nrn"].isin(self.train_nrn_lst)) & (axon_prepared_df[self.type_col].isin([2, 20, 21, 22, 23, 3, 30, 31, 32, 33]))]
        else:
            self.train = axon_prepared_df.loc[axon_prepared_df["nrn"].isin(self.train_nrn_lst)]
        self.test = axon_prepared_df.loc[axon_prepared_df["nrn"].isin(self.test_nrn_lst)]

        self.data_point = len(self.train)
        _label1 = len(self.train.loc[self.train["label"]==1])
        self.ration_of_label1 = _label1/self.data_point

        return


    def _run_model(self):

        ax0 = Classify_Axon(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.feature_type,
                            self.with_replacement, self.augment_nrn_lst, self.augment_number, self.train_only_axdn, self.seed, overwrite=self.overwrite)

        ax0 = ax0.classify_data()


        self.axon_classified_dict["info_dict"] = ax0.axon_classified_dict["info_dict"].copy()
        self.axon_classified_dict["sample_set"] = self.sample_set = ax0.axon_classified_dict["sample_set"].copy()

        if type(self.seed) is int:
            random.seed(self.seed)

        print("Run Axon Combination models...")
        for model_name, model in self.models.items():
            print(model_name,"...")

            test_dict = {self.feature_type: ax0.axon_classified_dict[model_name]["all"].copy()}

            if model_name == "svm":
                test_dict["feature_combinations"] = {self.feature_type: [self.feature_type]}
                self.axon_classified_dict[model_name] = test_dict
                continue

            top5_ftr = ax0.axon_classified_dict[model_name]["ftr_ranking"]
            top5_ftr = top5_ftr.head(5)["index"].tolist()
            ftr_comb_dict = feature_combinations(top5_ftr)

            test_dict["feature_combinations"] = ftr_comb_dict
            test_dict["feature_combinations"][self.feature_type] = [self.feature_type]
            test_df3 = None



            bar = progressbar.ProgressBar()
            time.sleep(0.01)
            for ftr_name, feature_lst in bar(ftr_comb_dict.items()):

                if ftr_name == self.feature_type:
                    continue


                for idx0, nrn_lst in enumerate(self.sample_set):
                    model1 = clone(model)

                    train_df = self.train.loc[~self.train['nrn'].isin(nrn_lst)].reset_index(drop=True)
                    test_df = self.test.loc[self.test['nrn'].isin(nrn_lst)].reset_index(drop=True)

                    y_test, y_pred, pred_proba, model1 = classification(train_df, test_df, label=['label'], features=feature_lst,
                                                                model=model1, pred_threshold=self.pred_prob,
                                                                standardization=False)


                    test_df2 = test_df[["nrn", "ID", "label"]].copy()
                    test_df2['inference'] = y_pred
                    test_df2['prob'] = pred_proba[:, 1]

                    if test_df3 is None:
                        test_df3 = test_df2
                    else:
                        test_df3 = test_df3.append(test_df2)

                    # # Append trained models
                    # trained_model.append(model1)


                test_df3 = test_df3.groupby(["nrn", self.child_col, "label"], as_index=False)["prob"].mean()
                test_dict[ftr_name] = test_df3



            self.axon_classified_dict[model_name] = test_dict

            time.sleep(0.01)

        return


    # csu todo
    def _evaluate(self):
        print("Evaluate models...")


        # 1.
        for m in list(self.models.keys()):
            print(m,'...')

            ftr_comb = self.axon_classified_dict[m]["feature_combinations"]

            df_eval = pd.DataFrame(index=list(ftr_comb.keys()), columns=["accuracy", "precision", "recall", "f1-score", "ftr_comb"])
            repredict_df_dict = {}
            cm_dict = {}
            wrong_dict = {}

            bar = progressbar.ProgressBar()
            time.sleep(0.01)
            for ftr_name, ftr_lst in bar(ftr_comb.items()):
                df_final = None
                w_lst = []
                df_a0 = self.axon_classified_dict[m][ftr_name]
                df_a = df_a0.copy()

                _lst0 = df_a["nrn"].unique().tolist()
                nrn_lst = sorted(_lst0)

                for idx0, nrn in enumerate(nrn_lst):
                    nrn0 = Data_Cleaner(input_folder, nrn)
                    n0 = nrn0.load_data()
                    pld = n0.polarity_dict
                    tnd = n0.tree_node_dict

                    df = n0.df[[self.child_col, self.parent_col, self.type_col]].copy()
                    df = add_bushCol(df, tnd, self.child_col, self.parent_col, descendant_only_forkLeaf=False)

                    a_pred = df_a.loc[(df_a["prob"] > 0.5) & (df_a["nrn"] == nrn), self.child_col].tolist()

                    repredict_dict = {"axon": [], "dendrite": []}
                    b0 = set(df.loc[df[self.child_col] == tnd["root"][0], "bush"].values[0])
                    while len(a_pred)>0:
                        a = a_pred[0]
                        a_bush = set(df.loc[df[self.child_col] == a, "bush"].values[0])
                        a_pred = list(set(a_pred) - a_bush)
                        ab0 = b0 & a_bush
                        b0 = b0 - ab0

                        repredict_dict["axon"] += list(ab0)

                    repredict_dict["dendrite"] += list(b0)

                    a_leaf = list(set(tnd["leaf"]) & set(pld["axon"]))
                    d_leaf = list(set(tnd["leaf"]) & set(pld["dendrite"]))
                    na_leaf = list(set(tnd["leaf"]) - set(a_leaf) - set(d_leaf))

                    df_re = df.loc[df[self.child_col].isin(a_leaf + d_leaf + na_leaf)].copy()
                    df_re["nrn"] = nrn
                    df_re = df_re[["nrn", self.child_col, self.type_col]].sort_values([self.child_col]).reset_index(
                        drop=True)
                    df_re["label"] = np.where(df_re[self.child_col].isin(a_leaf), 1, 0)
                    df_re["inference"] = np.where(df_re[self.child_col].isin(repredict_dict["axon"]), 1, 0)

                    if df_final is None:
                        df_final = df_re
                    else:
                        df_final = pd.concat([df_final, df_re])

                    _inference_set = set(df_re.loc[df_re["inference"]==1, self.child_col].tolist())
                    if set(a_leaf) != _inference_set:
                        w_lst.append(nrn)

                df_final = df_final.reset_index(drop=True)
                repredict_df_dict[ftr_name] = df_final


                # pred_prob = 0.5

                df_axon1 = df_final.copy()
                # df_axon1['inference'] = np.where(df_axon1['prob'] > pred_prob, 1, 0)
                y_test = df_axon1['label'].values
                y_pred = df_axon1['inference'].values

                df_eval.ix[ftr_name, "accuracy"] = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                df_eval.ix[ftr_name, "precision"] = report["1"]["precision"]
                df_eval.ix[ftr_name, "recall"] = report["1"]["recall"]
                df_eval.ix[ftr_name, "f1-score"] = report["1"]["f1-score"]
                df_eval.ix[ftr_name, "ftr_comb"] = ftr_lst

                cm = confusion_matrix(y_test, y_pred)
                cm_dict[ftr_name] = cm

                wrong_dict[ftr_name] = w_lst



            df_eval = df_eval.reset_index()
            df_eval = df_eval.rename(columns={"index": "ftr_name"})

            _df = None
            for i in range(5):
                i += 1
                _df0 = df_eval[df_eval['ftr_name'].str.startswith("P"+str(i))].copy().reset_index(drop=True)
                _df0 = _df0.loc[_df0['accuracy'] == _df0['accuracy'].max()]
                if _df is None:
                    _df = _df0
                else:
                    _df = _df.append(_df0)

            _df0 = df_eval.loc[df_eval['ftr_name'] == self.feature_type].copy().reset_index(drop=True)
            _df = _df.append(_df0)


            df_eval1 = _df.reset_index(drop=True)

            # self.axon_classified_dict[m]["confusion_matrix_highAcc"] = cm_dict[h_acc]
            self.axon_classified_dict[m]["repredict_df_dict"] = repredict_df_dict
            self.axon_classified_dict[m]["confusion_matrix_dict"] = cm_dict
            self.axon_classified_dict[m]["evaluation_df1"] = df_eval1
            self.axon_classified_dict[m]["evaluation_df1_all"] = df_eval
            self.axon_classified_dict[m]["wrong_prediction"] = wrong_dict
            # csu todo 4/3

            time.sleep(0.01)



            '''
            # 2.
            for m in bar(list(self.models.keys())):
                ftr_comb = self.axon_classified_dict[m]["feature_combinations"]

                df_eval = pd.DataFrame(index=list(ftr_comb.keys()),
                                       columns=["accuracy", "precision", "recall", "f1-score", "ftr_comb"])

                cm_dict = {}

                for ftr_name, ftr_lst in ftr_comb.items():
                    pred_prob = 0.5

                    df_axon = self.axon_classified_dict[m][ftr_name]
                    df_axon1 = df_axon.copy()
                    df_axon1['inference'] = np.where(df_axon1['prob'] > pred_prob, 1, 0)
                    y_test = df_axon1['label'].values
                    y_pred = df_axon1['inference'].values

                    df_eval.ix[ftr_name, "accuracy"] = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    df_eval.ix[ftr_name, "precision"] = report["1"]["precision"]
                    df_eval.ix[ftr_name, "recall"] = report["1"]["recall"]
                    df_eval.ix[ftr_name, "f1-score"] = report["1"]["f1-score"]
                    df_eval.ix[ftr_name, "ftr_comb"] = ftr_lst

                    cm = confusion_matrix(y_test, y_pred)
                    cm_dict[ftr_name] = cm

                df_eval = df_eval.reset_index()
                df_eval = df_eval.rename(columns={"index": "ftr_name"})

                _df = None
                for i in range(5):
                    i += 1
                    _df0 = df_eval[df_eval['ftr_name'].str.startswith("P" + str(i))].copy().reset_index(drop=True)
                    _df0 = _df0.loc[_df0['accuracy'] == _df0['accuracy'].max()]
                    if _df is None:
                        _df = _df0
                    else:
                        _df = _df.append(_df0)

                df_eval = _df.reset_index(drop=True)

                # self.axon_classified_dict[m]["confusion_matrix_highAcc"] = cm_dict[h_acc]
                self.axon_classified_dict[m]["confusion_matrix_dict"] = cm_dict
                self.axon_classified_dict[m]["evaluation_df1"] = df_eval
                '''

        # time.sleep(0.01)

        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.axon_classified_dict, file=file)

        return


    def _change(self):

        with open(self.fname, "wb") as file:
            pickle.dump(self.axon_classified_dict, file=file)

        return





if __name__ == '__main__':
    ax0 = Find_Best_Axon_Feature_Combination(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, feature_type, with_replacement, augment_nrn_lst, augment_number, train_with_only_axon_dendrite, seed, overwrite=overwrite)

    ax0 = ax0.classify_data()

    w_lst = ax0.axon_classified_dict["xgb"]["wrong_prediction"]["P4-2"]

    print(w_lst)
    print(len(w_lst))

    _lst = ["VGlut-F-800100","Gad1-F-800046","Cha-F-500046","Cha-F-100117","Gad1-F-600003","Gad1-F-400400","5-HT1B-F-500013","Cha-F-700121","Trh-F-300113","Gad1-F-100602","Cha-F-700121","VGlut-F-200012","VGlut-F-900093","VGlut-F-000600","Trh-F-400043"]

    print(set(w_lst).issuperset(set(_lst)))
    print(set(_lst)-set(w_lst))

    # ax0.evaluation_info()

    a = 123

    # ax0._load_data()
    #
    # ax0._evaluate()
    #
    # ax0._save_result()

    # ax_dict = ax0.axon_classified_dict
    # print(ax_dict["sample_set"])

    # ax0.evaluation_info()




########################################################################################################################
