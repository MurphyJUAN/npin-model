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
from nrn3_class_Prepare_Type import Prepare_Type



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
odd_nrn_type = ["multiple", "axonNear"]  # "all", "normal", "small", "multiple", or "axon_near"
remove_method = "leaf"
target_level = 5
with_replacement = True
seed = 123



########################################################################################################################
# Main Code
########################################################################################################################
class Classify_Type:

    def __init__(self,
                 input_folder,
                 odd_nrn_type,
                 remove_method=None,
                 target_level=None,
                 sampling_with_replacement=False,
                 seed=None,
                 pred_prob=None
                 ):


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
        self.seed = seed
        self.pred_prob = pred_prob


        # test samples
        self.sampling_times = 20
        self.sample_set = []
        _num_of_1 = self.num_of_1/len(neuron_dict["all"])
        _num_of_0 = self.num_of_0/len(neuron_dict["all"])
        if _num_of_1 < 0.2:
            _pct1 = 0.5
            _pct0 = (0.2*len(neuron_dict["all"]) - 0.5*len(self.odd_nrn_lst))/len(self.normal_nrn_lst)
            if not sampling_with_replacement:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0)
                for idx, s in enumerate(_set1):
                    self.sample_set.append(_set1[idx] + _set0[idx])
            else:
                _set1 , _ = list_sampling(self.odd_nrn_lst, pct=_pct1, with_replacement=True, sampling_times=self.sampling_times)
                _set0 , _ = list_sampling(self.normal_nrn_lst, pct=_pct0, with_replacement=True, sampling_times=self.sampling_times)
                for idx, s in enumerate(_set1):
                    self.sample_set.append(_set1[idx] + _set0[idx])

        elif _num_of_0 < 0.2:
            _pct1 = (0.2 * len(neuron_dict["all"]) - 0.5 * len(self.normal_nrn_lst)) / len(self.odd_nrn_lst)
            _pct0 = 0.5
            if not sampling_with_replacement:
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
            if not sampling_with_replacement:
                self.sample_set , _ = list_sampling(neuron_dict["all"], pct=_pct)
            else:
                self.sample_set , _ = list_sampling(neuron_dict["all"], pct=_pct, with_replacement=True, sampling_times=self.sampling_times)


        self.branch = 2
        self.feature_type = "f1"
        self.pred_prob_lst = np.arange(0.5, 1.05, 0.05)




        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["result", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["result", self.classify_target, _methodLevel])

        # _replaceTimes
        if sampling_with_replacement:
            _replaceTimes = "replace" + str(self.sampling_times)
            _fname1 = "_".join([_fname0, _replaceTimes])
        else:
            _fname1 = _fname0

        # _oddNrn
        _oddNrn = '&'.join(self.odd_nrn_type)
        _fname2 = "_".join([_fname1, _oddNrn])

        self.fname = input_folder + "nrn_result/" + _fname2 + ".pkl"


        # Models
        ridge_class = RidgeClassifier()
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True)
        xgb_class = XGBClassifier()
        self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class}


        return


    def data_info(self):
        info_dict = {"num_of_1": self.num_of_1, "num_of_0": self.num_of_0}

        print("num_of_1=%d," % info_dict["num_of_1"],
              "num_of_0=%d." % info_dict["num_of_0"])

        return info_dict

    
    
    def classify_data(self):
        if self._is_ready():
            self._load_data()
        else:
            self._load_data_from_Prepare_Type()
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


    def _load_data_from_Prepare_Type(self):
        tp0 = Prepare_Type(self.input_folder, self.remove_method, self.target_level)
        tp0 = tp0.load_data()
        type_prepared_df = tp0.type_prepared_df
        type_prepared_df["label"] = np.where(type_prepared_df["nrn"].isin(self.odd_nrn_lst), 1, 0)
        self.train = self.test = type_prepared_df

        return


    def _run_model(self):
        print("Run models...")

        if type(self.seed) is int:
            random.seed(self.seed)

        self.type_classified_dict = {}
        feature_dict = type_feature_dict(self.target_level, self.branch)
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

                test_df2 = test_df.copy()
                test_df2['inference'] = y_pred
                test_df2['prob'] = pred_proba[:, 1]
                test_dict[idx0] = test_df2

                if test_df3 is None:
                    test_df3 = test_df2
                else:
                    test_df3 = test_df3.append(test_df2)
                test_dict["all"] = test_df3

            self.type_classified_dict[model_name] = test_dict

        time.sleep(0.01)

        return


    def _evaluate(self):
        print("Evaluate models...")

        df_pred = pd.DataFrame(index=self.pred_prob_lst)

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for m in bar(list(self.models.keys())):

            for idx0, nrn_lst in enumerate(self.sample_set):
                df_tp = self.type_classified_dict[m][idx0]

                for pred_prob in self.pred_prob_lst:
                    df_tp1 = df_tp.copy()
                    df_tp1['inference'] = np.where(df_tp1['prob'] > pred_prob, 1, 0)
                    y_test = df_tp1['label'].values
                    y_pred = df_tp1['inference'].values

                    precision_lst = precision_score(y_test, y_pred, average=None)
                    precision = precision_lst[1]

                    col = "test_%d" % (idx0 + 1)
                    df_pred.set_value(pred_prob, col, precision)

            cols = list(df_pred)
            df_pred['min'] = df_pred.loc[:, cols[0]:cols[-1]].min(axis=1)
            df_pred['max'] = df_pred.loc[:, cols[0]:cols[-1]].max(axis=1)
            df_pred['average'] = df_pred.loc[:, cols[0]:cols[-1]].mean(axis=1)

            self.type_classified_dict[m]["precision_df"] = df_pred

        time.sleep(0.01)
        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.type_classified_dict, file=file)

        return





if __name__ == '__main__':
    tp0 = Classify_Type(input_folder, odd_nrn_type, remove_method, target_level, with_replacement, seed)

    tp0 = tp0.classify_data()
    
    tp_dict = tp0.type_classified_dict

    a = 123




########################################################################################################################
