# Purpose:
# 1. Create train & test set
# 2. Run model & Evaluate model
# 3. Outputs:
#       result_.pkl: train=test=, sampling the test & train set w/o replacement.
#       result_.pkl: train=test=, sampling w/ replacement for 20 times.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Prepare_Type_Known import Prepare_Type_Known



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
odd_nrn_type = ["small", "multiple", "axonNear"]  # "all", "normal", "small", "multiple", or "axon_near"
remove_method = "leaf"
target_level = 5
with_replacement = True
oversampling = True     # without synthesize
seed = 123



########################################################################################################################
# Main Code
########################################################################################################################
class Classify_Type_Known:

    def __init__(self,
                 input_folder,
                 odd_nrn_type,
                 remove_method=None,
                 target_level=None,
                 sampling_with_replacement=False,
                 oversampling = False,
                 seed=None,
                 pred_prob=None
                 ):

        self.type_classified_dict = {}


        self.odd_nrn_lst = []
        for i in odd_nrn_type:
            self.odd_nrn_lst += neuron_dict[i]
        self.normal_nrn_lst = [x for x in neuron_dict["all"] if x not in self.odd_nrn_lst]
        self.num_of_1 = len(self.odd_nrn_lst)
        self.num_of_0 = len(self.normal_nrn_lst)
        if any([self.num_of_1 == 0, self.num_of_0 == 0]):
            sys.exit("\n There is no neurons in odd_nrn or normal_nrn! Check Classify_Type.")


        self.classify_target = "type"
        self.input_folder = input_folder
        self.odd_nrn_type = sorted(odd_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.sampling_with_replacement = sampling_with_replacement
        self.oversampling = oversampling
        self.seed = seed
        self.pred_prob = pred_prob

        self.branch = 2
        self.feature_type = ["f1", "f2", "f3"]
        self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)



        # test samples
        self.sampling_times = 20
        self.sample_set = []
        self.ratio_of_label1 = self.num_of_1/len(neuron_dict["all"])
        self.ratio_of_label0 = self.num_of_0/len(neuron_dict["all"])
        if self.ratio_of_label1 < 0.2:
            _pct1 = 0.5
            _pct0 = (0.2*len(neuron_dict["all"]) - 0.5*len(self.odd_nrn_lst))/len(self.normal_nrn_lst)
            if not self.sampling_with_replacement:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0)
                for idx, s in enumerate(_set1):
                    self.sample_set.append(_set1[idx] + _set0[idx])
            else:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1, with_replacement=True, sampling_times=self.sampling_times)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0, with_replacement=True, sampling_times=self.sampling_times)
                for idx, s in enumerate(_set1):
                    self.sample_set.append(_set1[idx] + _set0[idx])

        elif self.ratio_of_label0 < 0.2:
            _pct1 = (0.2 * len(neuron_dict["all"]) - 0.5 * len(self.normal_nrn_lst)) / len(self.odd_nrn_lst)
            _pct0 = 0.5
            if not self.sampling_with_replacement:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0)
                for idx, s in enumerate(_set0):
                    self.sample_set[idx] = _set1[idx] + _set0[idx]
            else:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1, with_replacement=True, sampling_times=self.sampling_times)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0, with_replacement=True, sampling_times=self.sampling_times)
                for idx, s in enumerate(_set0):
                    self.sample_set[idx] = _set1[idx] + _set0[idx]

        else:
            _pct = 0.2
            if not self.sampling_with_replacement:
                self.sample_set , _ = list_sampling(neuron_dict["all"], pct=_pct)
            else:
                self.sample_set , _ = list_sampling(neuron_dict["all"], pct=_pct, with_replacement=True, sampling_times=self.sampling_times)

        self.type_classified_dict["sample_set"] = self.sample_set


        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["result", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["result", self.classify_target, _methodLevel])

        # _replaceTimes
        if self.sampling_with_replacement:
            _replaceTimes = "replace" + str(self.sampling_times)
            _fname1 = "_".join([_fname0, _replaceTimes])
        else:
            _fname1 = _fname0

        # oversampling
        if self.oversampling:
            _fname2 = "_".join([_fname1, "oversampling"])
        else:
            _fname2 = _fname1

        # _oddNrn
        _oddNrn = '&'.join(self.odd_nrn_type)
        _fname3 = "_".join([_fname2, _oddNrn, "Known"])

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
            for key, v0 in self.type_classified_dict.items():
                if key == "info_dict":
                    print_info_dict(self.type_classified_dict[key])
                    print("===========================================================================================")
                elif key != "sample_set":
                    print("Feature =", key, "\n")
                    for m, v1 in self.type_classified_dict[key].items():
                        print("Model =", m)
                        print("confusion_matrix:\n", self.type_classified_dict[key][m]["confusion_matrix"])
                        print("evaluation_df1:\n", self.type_classified_dict[key][m]["evaluation_df1"], "\n")
                    print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return

    
    
    def classify_data(self):
        if self._is_ready():
            self._load_data()
        else:
            self._load_data_from_Prepare_Type_Known()
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
            self.type_classified_dict = pickle.load(file)
        return


    def _load_data_from_Prepare_Type_Known(self):
        tp0 = Prepare_Type_Known(self.input_folder, self.remove_method, self.target_level)
        tp0 = tp0.load_data()
        type_prepared_df = tp0.type_prepared_df
        type_prepared_df["label"] = np.where(type_prepared_df["nrn"].isin(self.odd_nrn_lst), 1, 0)
        self.train = self.test = type_prepared_df

        return


    def _data_info(self):
        self.type_classified_dict["info_dict"] = {
            "target": self.classify_target,
            "odd_nrn_type": self.odd_nrn_type,
            "remove_method": self.remove_method,
            "target_level": self.target_level,
            "data_point": len(neuron_dict["all"]),
            "ratio_of_label1": round(self.ratio_of_label1, 2),
            "with_replacement": self.sampling_with_replacement,
            "oversampling": self.oversampling
        }

        return


    def _run_model(self):
        print("Run models...")

        if type(self.seed) is int:
            random.seed(self.seed)

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for f in bar(self.feature_type):
            feature_dict = type_feature_dict(self.target_level, self.branch)
            feature_lst = feature_dict[f]
            self.type_classified_dict[f] = {}

            for model_name, model in self.models.items():
                test_dict = {}
                test_df3 = None
                for idx0, nrn_lst in enumerate(self.sample_set):
                    train_df = self.train.loc[~self.train['nrn'].isin(nrn_lst)].reset_index(drop=True)
                    test_df = self.test.loc[self.test['nrn'].isin(nrn_lst)].reset_index(drop=True)


                    if all([self.oversampling, self.ratio_of_label1 < 0.2]):
                        df_small0 = train_df.loc[train_df["nrn"].isin(self. odd_nrn_lst)]
                        df_large = train_df.loc[train_df["nrn"].isin(self. normal_nrn_lst)]
                        _diff = len(df_large) - len(df_small0)
                        df_small1 = df_small0.sample(_diff, replace=True)
                        train_df = df_large.append(df_small1)
                        train_df = train_df.sample(frac=1).reset_index(drop=True)

                    elif all([self.oversampling, self.ratio_of_label0 < 0.2]):
                        df_small0 = train_df.loc[train_df["nrn"].isin(self.normal_nrn_lst)]
                        df_large = train_df.loc[train_df["nrn"].isin(self.odd_nrn_lst)]
                        _diff = len(df_large) - len(df_small0)
                        df_small1 = df_small0.sample(_diff, replace=True)
                        train_df = df_large.append(df_small1)
                        train_df = train_df.sample(frac=1).reset_index(drop=True)


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

                self.type_classified_dict[f][model_name] = test_dict

        time.sleep(0.01)

        return


    def _evaluate(self):
        print("Evaluate models...")

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for f in bar(self.feature_type):

            for m in list(self.models.keys()):
                df_eval = pd.DataFrame(index=self.pred_prob_lst)
                cm = None

                for pred_prob in self.pred_prob_lst:
                    acc_lst = []
                    prc_lst = []
                    rcl_lst = []
                    f1_lst  = []

                    for idx0, nrn_lst in enumerate(self.sample_set):
                        df_tp = self.type_classified_dict[f][m][idx0]
                        df_tp1 = df_tp.copy()
                        df_tp1['inference'] = np.where(df_tp1['prob'] > pred_prob, 1, 0)
                        y_test = df_tp1['label'].values
                        y_pred = df_tp1['inference'].values

                        acc_lst.append(accuracy_score(y_test, y_pred))
                        report = classification_report(y_test, y_pred, output_dict=True)
                        prc_lst.append(report["1"]["precision"])
                        rcl_lst.append(report["1"]["recall"])
                        f1_lst.append(report["1"]["f1-score"])

                        if pred_prob == 0.5:
                            cm0 = confusion_matrix(y_test, y_pred)
                            if cm is None:
                                cm = cm0
                            else:
                                cm += cm0

                    df_eval.set_value(pred_prob, "accuracy", np.mean(acc_lst))
                    df_eval.set_value(pred_prob, "precision", np.mean(prc_lst))
                    df_eval.set_value(pred_prob, "recall", np.mean(rcl_lst))
                    df_eval.set_value(pred_prob, "f1-score", np.mean(f1_lst))

                    if pred_prob == 0.5:
                        self.type_classified_dict[f][m]["confusion_matrix"] = cm

                self.type_classified_dict[f][m]["evaluation_df1"] = df_eval

        time.sleep(0.01)
        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.type_classified_dict, file=file)

        return





if __name__ == '__main__':
    tp0 = Classify_Type_Known(input_folder, odd_nrn_type, remove_method, target_level, with_replacement, oversampling, seed)

    # tp0 = tp0.classify_data()
    #
    # tp_dict = tp0.type_classified_dict

    tp0.evaluation_info()

    a = 123




########################################################################################################################
