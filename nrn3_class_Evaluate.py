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



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["normal"]  # "all", "normal", "small", "multiple", or "axonNear"
test_nrn_type = ["normal"]  # "all", "normal", "small", "multiple", or "axonNear"
remove_method = "leaf"
target_level = 5
with_replacement = True
seed=123

# train_nrn_type = ["normal"]  # "all", "normal", "small", "multiple", or "axonNear"
# test_nrn_type = ["axonNear"]  # "all", "normal", "small", "multiple", or "axonNear"
# remove_method = "leaf"
# target_level = 5
# with_replacement = True
# seed=123

# train_nrn_type = ["normal"]  # "all", "normal", "small", "multiple", or "axonNear"
# test_nrn_type = ["normal"]  # "all", "normal", "small", "multiple", or "axonNear"
# remove_method = "leaf"
# target_level = 5
# with_replacement = False
# seed=123


########################################################################################################################
# Main Code
########################################################################################################################
# csu todo: read in mlp result and evaluate
class Evaluate:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 sampling_with_replacement=True,
                 seed=None,
                 pred_prob=None,
                 overwrite=False
                 ):

        self.axon_classified_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.sampling_with_replacement = sampling_with_replacement
        self.seed = seed
        self.pred_prob = pred_prob
        self.overwrite = overwrite

        self.branch = 2
        self.sampling_pct = 0.2
        self.sampling_times = 20
        self.feature_type = "f1"
        self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)


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



        # test samples
        if all([self.train_nrn_type == self.test_nrn_type, not self.sampling_with_replacement]):
            self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct)

        elif all([self.train_nrn_type == self.test_nrn_type, self.sampling_with_replacement]):
            self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct, with_replacement=True, sampling_times=self.sampling_times)

        elif self.train_nrn_type != self.test_nrn_type:
            self.sample_set = [self.test_nrn_lst]

        self.axon_classified_dict["sample_set"] = self.sample_set


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["result", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["result", self.classify_target, _methodLevel])

        # _replaceTimes
        if all([self.train_nrn_type == self.test_nrn_type, self.sampling_with_replacement]):
            _replaceTimes = "replace" + str(self.sampling_times)
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

        self.fname = input_folder + "nrn_result/" + _fname2 + ".pkl"


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
            for key, v0 in self.axon_classified_dict.items():
                if key == "info_dict":
                    print_info_dict(self.axon_classified_dict[key])
                    print("===========================================================================================")
                elif key != "sample_set":
                    print("Model =", key)
                    print("confusion_matrix:\n", self.axon_classified_dict[key]["confusion_matrix"])
                    print("confusion_matrix (high precision):\n", self.axon_classified_dict[key]["confusion_matrix_highPrc"])
                    print("evaluation_df1:\n", self.axon_classified_dict[key]["evaluation_df1"], "\n")
                    print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return


    def classify_data(self):
        if self._is_ready():
            self._load_data()
        else:
            self._load_data_from_Prepare_Axon()
            self._data_info()
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


    def _data_info(self):
        self.axon_classified_dict["info_dict"] = {
            "target": self.classify_target,
            "train_nrn_type": self.train_nrn_type,
            "test_nrn_type": self.test_nrn_type,
            "remove_method": self.remove_method,
            "target_level": self.target_level,
            "data_point (train)": self.data_point,
            "ratio_of_label1 (train)": self.ration_of_label1,
            "with_replacement": self.sampling_with_replacement
        }

        return


    def _load_data_from_Prepare_Axon(self):
        ax0 = Prepare_Axon(self.input_folder, self.remove_method, self.target_level)
        ax0 = ax0.load_data()
        axon_prepared_df = ax0.axon_prepared_df
        self.train = axon_prepared_df.loc[axon_prepared_df["nrn"].isin(self.train_nrn_lst)]
        self.test = axon_prepared_df.loc[axon_prepared_df["nrn"].isin(self.test_nrn_lst)]
        
        self.data_point = len(self.train)
        _label1 = len(self.train.loc[self.train["label"]==1])
        self.ration_of_label1 = _label1/self.data_point

        return


    def _run_model(self):
        print("Run models...")

        if type(self.seed) is int:
            random.seed(self.seed)

        feature_dict = axmx_feature_dict(self.target_level, self.branch)
        feature_lst = feature_dict[self.feature_type]

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            test_dict = {}
            test_df3 = None
            for idx0, nrn_lst in enumerate(self.sample_set):
                train_df = self.train.loc[~self.train['nrn'].isin(nrn_lst)].reset_index(drop=True)
                test_df = self.test.loc[self.test['nrn'].isin(nrn_lst)].reset_index(drop=True)

                y_test, y_pred, pred_proba, model = classification(train_df, test_df, label=['label'], features=feature_lst,
                                                            model=model, pred_threshold=self.pred_prob,
                                                            standardization=False)

                test_df2 = test_df[["nrn", "ID", "label"]].copy()
                test_df2['inference'] = y_pred
                test_df2['prob'] = pred_proba[:, 1]
                test_dict[idx0] = test_df2

                if test_df3 is None:
                    test_df3 = test_df2
                else:
                    test_df3 = test_df3.append(test_df2)
                test_dict["all"] = test_df3

            self.axon_classified_dict[model_name] = test_dict

        time.sleep(0.01)

        return


    def _evaluate(self):
        print("Evaluate models...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for m in bar(list(self.models.keys())):
            df_eval = pd.DataFrame(index=self.pred_prob_lst)
            cm_lst = []

            for pred_prob in self.pred_prob_lst:
                acc_lst = []
                prc_lst = []
                rcl_lst = []
                f1_lst  = []
                cm = None

                for idx0, nrn_lst in enumerate(self.sample_set):
                    df_axon = self.axon_classified_dict[m][idx0]
                    df_axon1 = df_axon.copy()
                    df_axon1['inference'] = np.where(df_axon1['prob'] > pred_prob, 1, 0)
                    y_test = df_axon1['label'].values
                    y_pred = df_axon1['inference'].values

                    acc_lst.append(accuracy_score(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    prc_lst.append(report["1"]["precision"])
                    rcl_lst.append(report["1"]["recall"])
                    f1_lst.append(report["1"]["f1-score"])


                    cm0 = confusion_matrix(y_test, y_pred)
                    if cm is None:
                        cm = cm0
                    else:
                        cm += cm0

                df_eval.set_value(pred_prob, "accuracy", np.mean(acc_lst))
                df_eval.set_value(pred_prob, "precision", np.mean(prc_lst))
                df_eval.set_value(pred_prob, "recall", np.mean(rcl_lst))
                df_eval.set_value(pred_prob, "f1-score", np.mean(f1_lst))

                cm_lst.append(cm)

            _prc_lst = df_eval["precision"].tolist()
            h_prc = _prc_lst.index(max(_prc_lst))

            self.axon_classified_dict[m]["confusion_matrix"] = cm_lst[0]
            self.axon_classified_dict[m]["confusion_matrix_highPrc"] = cm_lst[h_prc]
            self.axon_classified_dict[m]["evaluation_df1"] = df_eval

        time.sleep(0.01)

        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.axon_classified_dict, file=file)

        return





if __name__ == '__main__':
    ax0 = Evaluate(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, with_replacement, seed)

    ax0 = ax0.classify_data()

    ax_dict = ax0.axon_classified_dict
    print(ax_dict["sample_set"])

    # ax0.evaluation_info()




########################################################################################################################
