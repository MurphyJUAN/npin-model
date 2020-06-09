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
from nrn3_class_Prepare_Mix import Prepare_Mix



########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
train_nrn_type = ["all"]  # "all", "normal", or "multiple"
test_nrn_type = ["all"]  # "all", "normal", or "multiple"
remove_method = "leaf"
target_level = 5
ax_feature_type = 'af8'
mx_feature_type = 'mf14'
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
class Classify_Mix:

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
                 overwrite=False
                 ):

        self.mix_classified_dict = {}
        self.classify_target = "mix"
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
        self.overwrite = overwrite

        self.branch = 2
        self.sampling_pct = 0.2
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



        # test samples
        if all([self.train_nrn_type == self.test_nrn_type, self.with_replacement is None]):
            self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct)

        elif all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
            target_num = len(self.train_nrn_lst)
            flat_num = 0
            while target_num != flat_num:
                self.sample_set , _ = list_sampling(self.train_nrn_lst, pct=self.sampling_pct, with_replacement=True, sampling_times=self.with_replacement)
                flat_lst = [item for sublist in self.sample_set for item in sublist]
                flat_lst = list_unique(flat_lst)
                flat_num = len(flat_lst)

        elif self.train_nrn_type != self.test_nrn_type:
            self.sample_set = [self.test_nrn_lst]
            self.with_replacement = None

        self.mix_classified_dict["sample_set"] = self.sample_set



        # fname
        # _methodLevel
        if all([remove_method is None, target_level is None]):
            _fname0 = "_".join(["result", self.classify_target])
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join(["result", self.classify_target, _methodLevel])

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
            for key, v0 in self.mix_classified_dict.items():
                if key == "info_dict":
                    print_info_dict(self.mix_classified_dict[key])
                    print("===========================================================================================")
                elif key != "sample_set":
                    print("Model =", key)
                    print("confusion_matrix:\n", turn_confusingMatrix_to_DF(self.mix_classified_dict[key]["confusion_matrix"]))
                    print("evaluation_df1:\n", self.mix_classified_dict[key]["evaluation_df1"], "\n")
                    print("evaluation_accuracy (by neuron):\n", self.mix_classified_dict[key]["evaluation_accuracy"], "\n")
                    print("===========================================================================================")


        else:
            self.classify_data()
            self.evaluation_info()

        return


    def classify_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._load_data_from_Prepare_Mix()
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
            self.mix_classified_dict = pickle.load(file)
        return


    def _data_info(self):
        self.mix_classified_dict["info_dict"] = {
            "target": self.classify_target,
            "train_nrn_type": self.train_nrn_type,
            "test_nrn_type": self.test_nrn_type,
            "remove_method": self.remove_method,
            "target_level": self.target_level,
            "ax_feature_type": self.ax_feature_type,
            "mx_feature_type": self.mx_feature_type,
            "data_point (train)": self.data_point,
            "ratio_of_label1 (train)": self.ration_of_label1,
            "sample_with_replacement_times": self.with_replacement
        }

        return


    def _load_data_from_Prepare_Mix(self):
        mx0 = Prepare_Mix(self.input_folder, self.train_nrn_type, self.test_nrn_type, self.remove_method, self.target_level, self.ax_feature_type, self.with_replacement)
        mx0 = mx0.load_data()
        mix_prepared_df = mx0.mix_prepared_df
        self.train = mix_prepared_df.loc[mix_prepared_df["nrn"].isin(self.train_nrn_lst)]
        self.test = mix_prepared_df.loc[mix_prepared_df["nrn"].isin(self.test_nrn_lst)]
        
        self.data_point = len(self.train)
        _label1 = len(self.train.loc[self.train["label"]==1])
        self.ration_of_label1 = _label1/self.data_point

        return


    def _run_model(self):
        print("Run Mix-point models...")

        if type(self.seed) is int:
            random.seed(self.seed)

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            feature_level = 5
            feature_dict = create_mix_feature_dict(feature_level, self.branch, model_name)
            feature_lst = feature_dict[self.mx_feature_type]

            test_dict = {}
            test_df3 = None
            trained_model = []
            if model_name != "svm":
                df_ftr_rank = pd.DataFrame(index=feature_lst)

            for idx0, nrn_lst in enumerate(self.sample_set):
                model1 = clone(model)

                train_df = self.train.loc[~self.train['nrn'].isin(nrn_lst)].reset_index(drop=True)
                test_df = self.test.loc[self.test['nrn'].isin(nrn_lst)].reset_index(drop=True)

                y_test, y_pred, pred_proba, model1 = classification(train_df, test_df, label=['label'], features=feature_lst, model=model1, pred_threshold=self.pred_prob, standardization=False)

                # 1. The result of each test
                test_df2 = test_df[["nrn", "ID", "label"]].copy()
                test_df2['inference'] = 0
                test_df2['prob'] = pred_proba[:, 1]

                # Find mix point form the prediction
                for nrn in nrn_lst:
                    _df = test_df2.loc[test_df2['nrn']==nrn].reset_index(drop=True)
                    _df, pred_lst = pred_mixPoint_from_prob(_df)
                    test_df2['inference'] = np.where((test_df2["nrn"]==nrn)&(test_df2["ID"].isin(pred_lst)), 1, test_df2['inference'])
                test_dict[idx0] = test_df2

                # Append the tests
                if test_df3 is None:
                    test_df3 = test_df2
                else:
                    test_df3 = test_df3.append(test_df2)

                # Append trained models
                trained_model.append(model1)

                # Add feature ranking columns
                if model_name != "svm":
                    df_ftr_rank[idx0] = model1.feature_importances_

            # Find top 10 features
            if model_name != "svm":
                df_ftr_rank['average'] = df_ftr_rank.mean(axis=1)
                df_ftr_rank = df_ftr_rank[['average']]
                df_ftr_rank = df_ftr_rank.sort_values(['average'], ascending=False).reset_index()
                df_ftr_rank = df_ftr_rank.head(10)

            # 2. Average result of all tests
            test_df3['inference'] = 0
            test_df3 = test_df3.groupby(["nrn", self.child_col, "label", 'inference'], as_index=False)["prob"].mean()

            # Predict mix-point form probability
            _nrn_lst = test_df3.nrn.unique()
            for nrn in _nrn_lst:
                _df = test_df3.loc[test_df3['nrn'] == nrn].reset_index(drop=True)
                _df, pred_lst = pred_mixPoint_from_prob(_df)
                test_df3['inference'] = np.where((test_df3["nrn"] == nrn) & (test_df3["ID"].isin(pred_lst)), 1, test_df3['inference'])
            test_dict["all"] = test_df3

            # Add trained models & ftr ranking to test_dict
            test_dict["trained_model"] = trained_model
            if model_name != "svm":
                test_dict["ftr_ranking"] = df_ftr_rank

            self.mix_classified_dict[model_name] = test_dict


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
            df_mix = self.mix_classified_dict[m]["all"]

            # 1. Evaluation DF and confusing matrix
            df_mix1 = df_mix.copy()
            y_test = df_mix1['label'].values
            y_pred = df_mix1['inference'].values

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
            nrn_lst = df_mix.nrn.unique()
            c_lst = []
            w_lst = []
            for nrn in nrn_lst:
                _df = df_mix.loc[df_mix['nrn']==nrn].reset_index(drop=True)
                _t = _df["label"].tolist()
                _p = _df["inference"].tolist()
                if _t == _p:
                    c_lst.append(nrn)
                else:
                    w_lst.append(nrn)
            c_ratio = len(c_lst) / len(nrn_lst)

            self.mix_classified_dict[m]["confusion_matrix"] = cm
            self.mix_classified_dict[m]["evaluation_df1"] = df_eval
            self.mix_classified_dict[m]["evaluation_accuracy"] = c_ratio
            self.mix_classified_dict[m]["correct_pred"] = c_lst
            self.mix_classified_dict[m]["wrong_pred"] = w_lst


            '''
            for idx0, nrn_lst in enumerate(self.sample_set):
                df_mix = self.mix_classified_dict[m][idx0]
                df_mix1 = df_mix.copy()
                y_test = df_mix1['label'].values
                y_pred = df_mix1['inference'].values

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
            
            df_eval["accuracy"] = np.mean(acc_lst)
            df_eval["precision"] = np.mean(prc_lst)
            df_eval["recall"] = np.mean(rcl_lst)
            df_eval["f1-score"] = np.mean(f1_lst)
            
            self.mix_classified_dict[m]["confusion_matrix"] = cm
            self.mix_classified_dict[m]["confusion_matrix_highPrc"] = cm_lst[h_prc]
            self.mix_classified_dict[m]["evaluation_df1"] = df_eval
            '''

        time.sleep(0.01)

        return


    def _save_result(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.mix_classified_dict, file=file)

        return





if __name__ == '__main__':
    for i in ['af9']:
        for j in ['mf4', 'mf8', 'mf9', 'mf10', 'mf11', 'mf14', 'mf15', 'mf16', 'mf17', 'mf18']:
            mx0 = Classify_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, i, j, with_replacement, seed, overwrite=overwrite)

            mx0 = mx0.classify_data()

            mx0.evaluation_info()
            print("===========================================================================================")


    # mx0 = Classify_Mix(input_folder, train_nrn_type, test_nrn_type, remove_method, target_level, ax_feature_type, mx_feature_type, with_replacement, seed, overwrite=overwrite)
    #
    # mx0 = mx0.classify_data()
    #
    # mx0.evaluation_info()




########################################################################################################################
