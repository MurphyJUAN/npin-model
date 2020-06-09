# Purpose:
# 1. Create train & test set
# 2. Run model & Evaluate model
# 3. Outputs:
#       result_axon_leaf5_normal.pkl: train=test=normal, sampling the test & train set w/o replacement.
#       result_axon_leaf5_replace20_normal.pkl: train=test=normal, sampling w/ replacement for 20 times.
#       result_axon_leaf5_normal&small.pkl: train=normal, test=small, no sampling.


########################################################################################################################
from .util import *
from .nrn_params import *
from .settings import *
from .nrn3_class_Prepare_Axon import Prepare_Axon
from .nrn3_class_DNN import DNN
from .nrn3_class_Data_Cleaner import Data_Cleaner
from .nrn3_class_Create_Axon_Feature import Create_Axon_Feature



########################################################################################################################
# Set up
########################################################################################################################
input_folder = '/data/'
# data
train_nrn_type = ["new_all_3"]  # "all", "normal", "small", "multiple", or "axonNear"
test_nrn_type = ["new_all_3"]  # "all", "normal", "small", "multiple", or "axonNear"
remove_method = None    # None or "leaf"
target_level = None     # None or int

# model
models = ["xgb"]      # "svm", "gbdt", "rf", "ridge", "xgb", "dnn"
mdl_ensemble = 2                 # int >= 1
features = ['s', 'ns', 'ds', 'nds', 'l', 'nl', 'c1', 'as', 'rc']    # 'l', 'nl', 's', 'ns', 'ds', 'nds', 'ro', 'c', 'rc'
pyramid_layer = 1       # int >= 1, 1: only target node, 2: target node + 1 generation, 3: target node + 2 generation.
threshold_layer = 1     # int >= 1, 1: need no descendants, 2: need at least 1 generation
sample = "region"   # "random" or "region"
sample_pct = 0.585    # (0, 1) or None: use all "train_nrn_type" data to train
sample_with_replacement = 20    # int >= 1

# data augmentation
augment_nrn_type = ["new_all_3"]
augment_number = 0     # int >= 0

# other
overwrite = False    # True: overwrite the predict result of classify_data(), evaluation of evaluate()
rerun_mdl = False    # True: not use trained dnn model to predict
relabel = True       # True: use post_relabel to evaluate
show = True          # True: show evaluation result

only_terminal=True   # True: plot terminal only
branch_col="l"       # None: show no branch length; "l": show branch length
show_node_id=True    # True: plot nodes with id number


########################################################################################################################
# Main Code
########################################################################################################################
class Classify_Axon:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 models=["xgb", "dnn"],
                 mdl_ensemble=1,
                 features=['l', 'nl', 's', 'ns', 'ds', 'nds', 'ro', 'c', 'rc'],
                 pyramid_layer=6,
                 threshold_layer=4,
                 sample="random",
                 sample_pct=None,
                 sample_with_replacement=10,
                 augment_nrn_type=None,
                 augment_number=0,
                 child_col='ID',
                 parent_col="PARENT_ID",
                 type_col='type_pre',   # Here use pre_relabel type 'type_pre'; the original type is 'T'.
                 overwrite=False
                 ):
        self.sys_path = os.path.dirname(__file__)
        self.axon_classified_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.features = sorted(features)
        self.pyr_layer = pyramid_layer
        self.th_layer = threshold_layer
        self.sample = sample
        self.sample_pct = sample_pct
        self.with_replacement = sample_with_replacement
        self.augment_nrn_type = augment_nrn_type
        self.augment_number = augment_number
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite


        # Other configs
        self.branch = 2     # remove redundant branch (> 2)
        if sample_pct is not None:
            '''
            select a portion of data to test set(when use random) or train set(use region)
            '''
            self.sample_pct = sample_pct


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
        self.train_nrn_lst = sorted(self.train_nrn_lst)
        self.test_nrn_lst = sorted(self.test_nrn_lst)

        self.test_set = [self.test_nrn_lst for i in range(self.with_replacement)]
        self.train_set = [self.train_nrn_lst for i in range(self.with_replacement)]
        # test & train samples
        if set(self.train_nrn_lst).isdisjoint(set(self.test_nrn_lst)):
            self.test_set = [self.test_nrn_lst for i in range(self.with_replacement)]
            self.train_set = [self.train_nrn_lst for i in range(self.with_replacement)]
            # self.with_replacement = None

        elif sample == "random":
            '''
            # this is for paper exp
            if train_nrn_type == ["Blowfly"]:
                self.test_set = b_random_20_test_4
                self.train_set = b_random_20_train_15
            elif train_nrn_type == ["Moth"]:
                self.test_set = m_random_20_test_1
                self.train_set = m_random_20_train_2
            elif train_nrn_type == ["simple"]:
                self.test_set = s_random_20_test_18
                self.train_set = s_random_20_train_71
            elif train_nrn_type == ["complex"]:
                self.test_set = c_random_20_test_25
                self.train_set = c_random_20_train_99
            elif train_nrn_type == ["axon_near"]:
                self.test_set = a_random_20_test_3
                self.train_set = a_random_20_train_13
            elif self.with_replacement==50:
                self.test_set = random_50_test_50
                self.train_set = []
                if self.sample_pct==0.345:
                    self.train_set = random_50_train_150
                elif self.sample_pct==0.452:
                    self.train_set = random_50_train_125
                elif self.sample_pct==0.562:
                    self.train_set = random_50_train_100
                elif self.sample_pct==0.672:
                    self.train_set = random_50_train_75
            elif self.with_replacement==10:
                self.test_set = random_10_test_50
                self.train_set = []
                if self.sample_pct == 0.452:
                    self.train_set = random_10_train_125
            '''

            if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
                self.test_set, self.train_set = list_sampling(self.train_nrn_lst, pct=self.sample_pct, with_replacement=True, sampling_times=self.with_replacement)


        elif sample == "region":
            '''
            # this is for paper exp
                        if self.with_replacement==50:
                self.test_set = random_50_test_50
                self.train_set = []
                if self.sample_pct == 0.54:
                    self.train_set = region_50_train_125
            elif self.with_replacement==10:
                self.test_set = random_10_test_50
                self.train_set = []
                if self.sample_pct == 0.54:
                    self.train_set = region_10_train_125
            elif self.with_replacement == 20:
                self.test_set = random_20_test_50
                self.train_set = []
                if self.sample_pct == 0.585:
                    self.train_set = region_20_train_125
            '''

            # dict samples
            if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
                self.train_set, self.test_set = dict_sampling(group_dict, pct=self.sample_pct, sample_times=self.with_replacement, min=1)


        self.axon_classified_dict["sample_test_set"] = self.test_set
        self.axon_classified_dict["sample_train_set"] = self.train_set



        # Create fake nrn lst
        self.fk_nrn_name_lst = []
        augment_nrn_lst = []
        if all([isinstance(augment_nrn_type, list), augment_number > 0]):
            for t in augment_nrn_type:
                augment_nrn_lst += neuron_dict[t]
            for nrn_name in augment_nrn_lst:
                self.fk_nrn_name_lst += ["fk" + str(i) + "_" + nrn_name for i in range(augment_number)]




        # fname
        _fname0 = "_".join(["result", self.classify_target])

        # _trainTest
        if self.train_nrn_type == self.test_nrn_type:
            _train = _test = _trainTest = '&'.join(self.train_nrn_type)
            _fname0 = "_".join([_fname0, _trainTest])
        else:
            _train = '&'.join(self.train_nrn_type)
            _test = '&'.join(self.test_nrn_type)
            _trainTest = "_".join([_train, _test])
            _fname0 = "_".join([_fname0, _trainTest])

        # _methodLevel
        if all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            _fname0 = "_".join([_fname0, _methodLevel])

        # _features
        _features = "-".join(self.features)
        _fname0 = "_".join([_fname0, _features])

        # _pyr_th
        _pyr_th = "".join(["pyr", str(self.pyr_layer), "th", str(self.th_layer)])
        _fname0 = "_".join([_fname0, _pyr_th])

        # _sample_time_pct
        if type(self.with_replacement) is int:
            _sample_time_pct = sample + str(self.with_replacement) + "pct" + str(self.sample_pct)
            _fname0 = "_".join([_fname0, _sample_time_pct])

        # _augment
        if any([augment_nrn_type is None, augment_number == 0]):
            pass
        elif all([isinstance(augment_nrn_type, list), type(augment_number) is int]):
            _augment = "fk" + str(augment_number)
            _fname0 = "_".join([_fname0, _augment])
        else:
            sys.exit("\n augment_nrn_type = None or list; augment_number = int! Check Prepare_Axon.")


        # result fname
        self.fname_dict = {}
        self.mdl_ensemble = {}
        for m in models:
            # _mdl_ensemble (dnn model only)
            if all([m=="dnn", mdl_ensemble > 1]):
                self.mdl_ensemble[m] = mdl_ensemble
                _mdl_ensemble = "".join(["ens", str(mdl_ensemble)])
                _m = "_".join([m, _mdl_ensemble])
            elif mdl_ensemble > 1:
                print("\n Now only dnn can be used with ensemble. Mld_ensemble of other model will be set as default 1.")
                self.mdl_ensemble[m] = 1
                _m = m
            else:
                self.mdl_ensemble[m] = 1
                _m = m

            self.fname_dict[m] = input_folder + "nrn_result/" + "_".join([_fname0, _m]) + ".pkl"


        # model fname
        self.model_fname_dict = {}
        for m in models:
            self.model_fname_dict[m] = {}
            mfname = _fname0.replace("result", "model")
            mfname = "_".join([mfname, m])
            if all([self.train_nrn_type == self.test_nrn_type, type(self.with_replacement) is int]):
                for i in range(self.with_replacement):
                    self.model_fname_dict[m][i] = input_folder + "nrn_model/" + "_".join([mfname, str(i)])
            else:
                self.model_fname_dict[m][0] = input_folder + "nrn_model/" + "_".join([mfname, str(0)])

        # trained model fname
        self.trained_model_fname_dict = {}
        for m in models:
            self.trained_model_fname_dict[m] = {}
            mfname = _fname0.replace("result", "model")
            mfname = mfname.replace(_trainTest, _train)
            mfname = "_".join([mfname, m])
            if type(self.with_replacement) is int:
                for i in range(self.with_replacement):
                    self.trained_model_fname_dict[m][i] = input_folder + "nrn_model/" + "_".join([mfname, str(i)])
            else:
                self.trained_model_fname_dict[m][0] = input_folder + "nrn_model/" + "_".join([mfname, str(0)])


        # Models
        gbdt_class = ensemble.GradientBoostingClassifier()
        rf_class = ensemble.RandomForestClassifier()
        svm_class = SVC(kernel='rbf', class_weight='balanced', probability=True, gamma="scale")
        xgb_class = XGBClassifier()
        # lgb_class = lgb.LGBMClassifier()
        dnn_class = DNN([5, 5], "sigmoid", 300, 300, '')  # best
        # dnn_class = DNN([10, 10], "selu", 1500, 600, '')
        # dnn_class = DNN([5, 5], "selu", 10, 10, '')   # test

        self.models = {'gbdt': gbdt_class, 'rf': rf_class, 'svm': svm_class, 'xgb': xgb_class, 'dnn':dnn_class}
        self.models = {k: self.models[k] for k in models}

        return



    def classify_data(self, rerun_mdl=False):
        self.rerun_mdl = rerun_mdl
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._prepare_data_from_Create_Axon_Feature()
            self._run_model()
            self._load_data()
        return self


    def predict(self, prob_threshold=None):
        self.prob_threshold = prob_threshold
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._prepare_data_from_Create_Axon_Feature()
            self._predict(prob_threshold)
            self._load_data()
        return self


    def evaluate(self, relabel=True, show=True, prob_thresholds=np.arange(0.5, 1, 0.05)):
        self.post_relabel = relabel
        self.prob_thresholds = prob_thresholds
        if not self.post_relabel:
            self.prob_thresholds = [0.5]    # if not post relabel, pred threshold will fixed at 0.5
        if not self._is_ready():
            self.classify_data()
        self._load_data()
        self._data_info()
        self._evaluate()
        if show:
            self._show_evaluation()
        return


    def plot_result(self, prob_threshold, file_type, file_size='20, 20', only_terminal=True, branch_col="l", show_node_id=True):
        self.prob_threshold = prob_threshold
        self.file_type = file_type
        self.file_size = file_size
        self.only_terminal = only_terminal
        self.branch_col = branch_col
        self.show_node_id = show_node_id
        if not self._is_ready():
            self.classify_data()
        self._load_data()
        self._plot_result()
        return


    def load_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        return self.axon_classified_dict


    def _is_ready(self):
        for _, fname in self.fname_dict.items():
            if os.path.exists(fname):
                x = True
            else:
                x = False
                break
        return x


    def _load_data(self):
        self.axon_classified_dict = {}
        for k, v in self.fname_dict.items():
            with open(self.sys_path + v, "rb") as file:
                self.axon_classified_dict[k] = pickle.load(file)
        return


    def _prepare_data_from_Create_Axon_Feature(self):
        print("Prepare Data...")
        # train
        ax0 = Prepare_Axon(self.input_folder, self.train_nrn_type, self.remove_method, self.target_level, self.augment_number)
        ax0 = ax0.load_data(["forTrain", "forTrainFk"])
        df = ax0.prepare_dict["forTrain"]
        if self.augment_number > 0:
            fk = ax0.prepare_dict["forTrainFk"]
            fk = fk.loc[fk["nrn"].isin(self.fk_nrn_name_lst)]
            _df = df.append(fk)
        else:
            _df = df.copy()
        del ax0

        # test
        if self.train_nrn_type != self.test_nrn_type:
            ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level, self.augment_number)
            ax0 = ax0.load_data(["forTrain"])
            df = ax0.prepare_dict["forTrain"]
            del ax0

        gc.collect()

        self.test = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=False, empty_number=-1)
        self.train = pyramid(_df, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)
        a=123
        # with open("./data/nrn_cleaned/prepare_axon", "rb") as file:
        #     df = pickle.load(file)
        #
        # with open("./data/nrn_cleaned/prepare_fk", "rb") as file:
        #     fk = pickle.load(file)
        #     fk = fk.loc[fk["nrn"].isin(self.fk_nrn_name_lst)]
        #     _df = df.append(fk)
        #
        # self.test = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=False, empty_number=-1)
        # self.train = pyramid(_df, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)

        # start_time = time.time()
        # # test data
        # df = None
        # for nrn in self.train_nrn_lst:
        #     n1 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level)
        #     n1 = n1.load_data()
        #     _df = n1.df_axon
        #     if df is None:
        #         df = _df
        #     else:
        #         df = df.append(_df)
        #     del n1, _df
        #     gc.collect()
        # df = df.sort_values(['nrn', 'ID']).reset_index(drop=True)
        # self.test = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=False, empty_number=-1)
        #
        # # train data
        # if self.train_nrn_type == self.test_nrn_type:
        #     self.train = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)
        #
        #     # Add fake nrn
        #     if self.fk_nrn_name_lst:
        #         df = None
        #         for nrn in self.fk_nrn_name_lst:
        #             n1 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level)
        #             n1 = n1.load_data()
        #             _df = n1.df_axon
        #             if df is None:
        #                 df = _df
        #             else:
        #                 df = df.append(_df)
        #             del n1, _df
        #             gc.collect()
        #         df = df.sort_values(['nrn', 'ID']).reset_index(drop=True)
        #         fk_train = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)
        #         self.train = self.train.append(fk_train)
        #
        # else:
        #     total_train_nrn_lst = list_unique(self.fk_nrn_name_lst + self.train_nrn_lst)
        #     df = None
        #     for nrn in total_train_nrn_lst:
        #         n1 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level)
        #         n1 = n1.load_data()
        #         _df = n1.df_axon
        #         if df is None:
        #             df = _df
        #         else:
        #             df = df.append(_df)
        #         del n1, _df
        #         gc.collect()
        #     df = df.sort_values(['nrn', 'ID']).reset_index(drop=True)
        #     self.train = pyramid(df, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)
        #
        # print("Elapsed Time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
        return


    def _run_model(self):
        print("Run Axon Models...")

        # csu todo: remove while pyramid complete
        feature_lst = list(set(self.test.columns) - set(["nrn", "ID"]))

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            # check if result exist
            if all([os.path.exists(self.fname_dict[model_name]), not self.overwrite]):
                continue

            if self.mdl_ensemble[model_name] == 1:
                random.seed(123)

            if model_name not in ('svm', 'dnn'):
                df_ftr_rank = pd.DataFrame(index=feature_lst)

            test_df3 = None
            for idx0 in range(self.with_replacement):
                # test df
                test_df = self.test.loc[self.test['nrn'].isin(self.test_set[idx0])].reset_index(drop=True)

                # train df
                _nrn_lst = self.train_set[idx0]
                _fk_lst = []
                if self.fk_nrn_name_lst:
                    for nrn_name in _nrn_lst:
                        _fk_lst += ["fk" + str(i) + "_" + nrn_name for i in range(self.augment_number)]
                else:
                    pass
                _nrn_lst += _fk_lst
                train_df = self.train.loc[self.train['nrn'].isin(_nrn_lst)].reset_index(drop=True)


                # run model
                for e in range(self.mdl_ensemble[model_name]):
                    # find save path
                    if model_name == "dnn":
                        sp = self.model_fname_dict[model_name][idx0] + "-" + str(e) + ".pkl"
                    elif model_name == "xgb":
                        sp = self.model_fname_dict[model_name][idx0] + "-" + str(e) + ".model"

                    # use exist models to predict
                    if all([os.path.exists(sp), not self.rerun_mdl]):
                        if model_name == "dnn":
                            model1.save_dir(sp)
                        elif model_name == "xgb":
                            model1 = XGBClassifier()  # init model
                            model1.load_model(sp)  # load data
                        pred_proba = model1.predict_proba(np.array(test_df[feature_lst]))
                    # run new models
                    else:
                        if model_name == "dnn":
                            model.save_dir(sp)
                        y_test, y_pred, pred_proba, model1 = classification(train_df, test_df, label=['label'], features=feature_lst, model=model)
                        if model_name =="xgb":
                            model1.save_model(sp)


                    test_df2 = test_df[['nrn', 'ID']].copy()
                    if model_name == "dnn":
                        test_df2['prob'] = pred_proba
                    else:
                        test_df2['prob'] = pred_proba[:, 1]

                    if test_df3 is None:
                        test_df3 = test_df2
                    else:
                        test_df3 = test_df3.append(test_df2)

                    # Add feature ranking columns
                    if model_name not in ('svm', 'dnn'):
                        df_ftr_rank[idx0] = model1.feature_importances_

                    del model1, sp

            # Find top 10 features
            if model_name not in ('svm', 'dnn'):
                df_ftr_rank['average'] = df_ftr_rank.mean(axis=1)
                df_ftr_rank = df_ftr_rank[['average']]
                df_ftr_rank = df_ftr_rank.sort_values(['average'], ascending=False).reset_index()
                df_ftr_rank = df_ftr_rank.head(10)
                # print(df_ftr_rank)

            test_df3 = test_df3.groupby(["nrn", self.child_col], as_index=False)["prob"].mean()
            self.axon_classified_dict["avg_result"] = test_df3.sort_values(['nrn', 'ID']).reset_index(drop=True)

            # Add ftr ranking
            if model_name not in ('svm', 'dnn'):
                self.axon_classified_dict["ftr_ranking"] = df_ftr_rank

            # save result
            with open(self.fname_dict[model_name], "wb") as file:
                pickle.dump(self.axon_classified_dict, file=file)

        time.sleep(0.01)

        return


    def _predict(self, prob_threshold):
        print("Predict...")
        sys_path = os.path.dirname(__file__)
        feature_lst = list(set(self.test.columns) - set(["nrn", "ID"]))

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for model_name, model in bar(self.models.items()):
            # check if result exist
            if all([os.path.exists(self.fname_dict[model_name]), not self.overwrite]):
                continue

            test_df3 = None
            for idx0 in range(self.with_replacement):
                # test df
                test_df = self.test.loc[self.test['nrn'].isin(self.test_set[idx0])].reset_index(drop=True)

                # run model
                for e in range(self.mdl_ensemble[model_name]):
                    # find saved model path
                    if model_name == "dnn":
                        sp = sys_path + self.trained_model_fname_dict[model_name][idx0] + "-" + str(e) + ".pkl"
                    elif model_name == "xgb":
                        sp = sys_path + self.trained_model_fname_dict[model_name][idx0] + "-" + str(e) + ".model"
                    # use exist models to predict
                    if os.path.exists(sp):
                        if model_name == "dnn":
                            model1.save_dir(sp)
                        elif model_name == "xgb":
                            model1 = XGBClassifier()  # init model
                            model1.load_model(sp)  # load trained parameters
                        pred_proba = model1.predict_proba(np.array(test_df[feature_lst]))
                    else:
                        print("No trained model available for prediction.")


                    test_df2 = test_df[['nrn', 'ID']].copy()
                    if model_name == "dnn":
                        test_df2['prob'] = pred_proba
                    else:
                        test_df2['prob'] = pred_proba[:, 1]

                    if test_df3 is None:
                        test_df3 = test_df2
                    else:
                        test_df3 = test_df3.append(test_df2)

                    del model1, sp


            test_df3 = test_df3.groupby(["nrn", self.child_col], as_index=False)["prob"].mean()
            self.axon_classified_dict["avg_result"] = test_df3.sort_values(['nrn', 'ID']).reset_index(drop=True)

            # save result
            with open(sys_path + self.fname_dict[model_name], "wb") as file:
                pickle.dump(self.axon_classified_dict, file=file)

        time.sleep(0.01)

        # cj todo: Post-relabel
        '''
        0. (done)open a new branch for this issue
        1. (done)add new empty folder "nrn_pred" to "./data" & also add it to .gitignore.
        2. (done)use prob_threshold as your logic gate (if prob_threshold=None then ignore; if 0<prob_threshold<1 then use post-relabel & save).
        4. (done)create the neuron list and loop over the list: 1. add post relabel columns 2. save new swc to new folder "./data/nrn_pred".
        5. put dnn result behind xgb
        '''
        #post relabel
        if prob_threshold == None :
            print("No probability threshold.")
        elif prob_threshold != None :
            df = self.axon_classified_dict["avg_result"]
            df = post_relabel(df, threshold=self.prob_threshold)
            if len(list(df['type_post'])) == 0 :
                self._predict(prob_threshold)
            elif max(list(df['type_post'])) == min(list(df['type_post'])) :
                self._predict(prob_threshold)
            else :
                pass

        # save result
        with open(sys_path + self.fname_dict[model_name], "wb") as file:
            pickle.dump(self.axon_classified_dict, file=file)

        #save result into nrn_pred
        for test_swc_name in self.test_nrn_lst :
            if not os.path.exists(self.input_folder + "nrn_pred/" + test_swc_name + "_predicted" + ".swc"):
                fname_swc = sys_path + self.input_folder + "nrn_original/" + test_swc_name + ".swc"
                nrn = nm.io.swc.read(fname_swc)
                origin_cols = ['X','Y','Z','D/2','R','ID','PARENT_ID']
                df0 = pd.DataFrame(nrn.data_block,columns=origin_cols)
                df1 = df.drop(['nrn','prob'],axis=1)

                for i in range(len(df)) :
                    if df['nrn'][i] != test_swc_name :
                        df1=df1.drop(index=[i])

                #replace R with result : for network
                df0 = df0.drop(['R'], axis=1)
                df1 = df1.rename(columns={'type_post': 'R'})
                df2 = pd.merge(df0, df1, on='ID', how='outer')
                df2 = df2.loc[:,['ID','R','X','Y','Z','D/2','PARENT_ID']]

                # if model_name == 'xgb' :
                #     df1=df1.rename(columns={'type_post':'type_xgb'})
                #     df2 = pd.merge(df0,df1,on='ID',how='outer')
                    # df0['type_xgb'] = df['type_post']
                # elif model_name == 'dnn' :
                #     df1=df1.rename(columns={'type_post': 'type_dnn'})
                #     df2 = pd.merge(df0, df1, on='ID',how='outer')
                    # df0['type_dnn'] = df['type_post']

                df2 = df2.fillna(1, limit=1)
                df2 = df2.fillna(0)
                # df2.index = df2.index + 1
                df2[['ID','PARENT_ID','R']] = df2[['ID','PARENT_ID','R']].astype(int)

            else :
                pre_f = pd.read_csv(self.input_folder + "nrn_pred/" + test_swc_name + "_predicted" + ".swc",
                                    sep=' ',index_col=None,header=0)
                df0 = pd.DataFrame(pre_f)
                df1 = df.drop(['nrn', 'prob'], axis=1)

                for i in range(len(df)) :
                    if df['nrn'][i] != test_swc_name :
                        df1=df1.drop(index=[i])

                # replace R with result : for network
                df0 = df0.drop(['R'],axis=1)
                df1 = df1.rename(columns={'type_post': 'R'})
                df2 = pd.merge(df0, df1, on='ID', how='outer')
                df2 = df2.loc[:, ['ID', 'R', 'X', 'Y', 'Z', 'D/2', 'PARENT_ID']]

                # if model_name == 'xgb' :
                #     if 'type_xgb' in df0.columns :
                #         df0 = df0.drop(['type_xgb'],axis=1)
                #     else :
                #         pass
                #     df1 = df1.rename(columns={'type_post':'type_xgb'})
                #     df2 = pd.merge(df0,df1,on='ID',how='outer')
                    # df0['type_xgb'] = df['type_post']
                # elif model_name == 'dnn' :
                #     if 'type_dnn' in df0.columns :
                #         df0 = df0.drop(['type_dnn'],axis=1)
                #     else :
                #         pass
                #     df1 = df1.rename(columns={'type_post': 'type_dnn'})
                #     df2 = pd.merge(df0, df1, on='ID',how='outer')
                    # df0['type_dnn'] = df['type_post']

                df2 = df2.fillna(1, limit=1)
                df2 = df2.fillna(0)
                # df2.index=df2.index+1
                df2[['ID','PARENT_ID','R']] = df2[['ID','PARENT_ID','R']].astype(int)

            # save as swc
            # df0.to_csv(path_or_buf=self.input_folder+'nrn_pred/'+test_swc_name+"_predicted_"+model_name+".swc",
            #            na_rep='be cut',sep=' ')
            df2.to_csv(path_or_buf=sys_path + self.input_folder + 'nrn_pred/' + test_swc_name + "_predicted" + ".swc",
                       na_rep='be cut', sep=' ',index=None)

            #save as csv
            # df0.to_csv(path_or_buf=self.input_folder+'nrn_pred/'+test_swc_name+"_predicted_"+model_name+".csv",
            #            na_rep='be cut')
            df2.to_csv(path_or_buf=sys_path + self.input_folder + 'nrn_pred/' + test_swc_name + "_predicted" + ".csv",
                       na_rep='be cut',index=None)

        return


    def _data_info(self):
        self.axon_classified_dict["info_dict"] = {
            "target": self.classify_target,
            "train_nrn_type": self.train_nrn_type,
            "test_nrn_type": self.test_nrn_type,
            "remove_method": self.remove_method,
            "target_level": self.target_level,
            "mdl_ensemble(only for dnn)": self.mdl_ensemble,
            "features": self.features,
            "pyramid_layer": self.pyr_layer,
            "threshold_layer":self.th_layer,
            "sample": self.sample,
            "sample_with_replacement_times": self.with_replacement,
            "train_nrn_num": len(self.train_set[0]),
            "test_nrn_num": len(self.test_set[0]),
            "augment_nrn_type": self.augment_nrn_type,
            "augment_number": self.augment_number,
            "post_relabel": self.post_relabel
        }

        return


    def _evaluate(self):
        print("Evaluate models...")

        df0 = None
        bar = progressbar.ProgressBar()
        time.sleep(0.01)

        # 1.
        for m in bar(list(self.models.keys())):

            try:
                self.axon_classified_dict[m]["evaluation_df1"]
            except:
                pass
            else:
                if all([self.post_relabel, not self.overwrite]):
                    continue

            # Full "original" level tree
            if df0 is None:

                ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level)
                ax0 = ax0.load_data(["forEvaluate"])
                df0 = ax0.prepare_dict["forEvaluate"]
                del ax0
                gc.collect()


                # with open("./data/nrn_cleaned/prepare_all", "rb") as file:
                #     df0 = pickle.load(file)


                # for nrn in self.test_nrn_lst:
                #     n1 = Data_Cleaner(self.input_folder, nrn, self.remove_method, self.target_level)
                #     n1 = n1.load_data()
                #     _df = update_parent_col(n1.df, n1.df_dis, n1.tree_node_dict, self.child_col, self.parent_col)
                #     if df0 is None:
                #         df0 = _df
                #     else:
                #         df0 = df0.append(_df)
                #     del _df, n1
                #     gc.collect()
                # df0 = df0.sort_values(['nrn', 'ID']).reset_index(drop=True)
                # df0 = df0.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'type_pre']]
            else:
                pass

            cm_lst = []
            acc_lst = []
            df_pr = pd.DataFrame(index=["axon", "dend"], columns=["prc", "rcl"])

            # axon
            df_eval = pd.DataFrame(index=self.prob_thresholds)
            prc_lst = []
            rcl_lst = []
            f1_lst = []

            # dendrite
            df_eval0 = pd.DataFrame(index=self.prob_thresholds)
            prc_lst0 = []
            rcl_lst0 = []
            f1_lst0 = []


            for pred_prob in self.prob_thresholds:
                df = self.axon_classified_dict[m]["avg_result"]
                df = pd.merge(df0, df, how="left", on=['nrn', 'ID'])
                # df = df.loc[~df["nrn"].isin(group_dict["Type_LOB_OG"])]

                # post_relabel
                if self.post_relabel:
                    df = post_relabel(df, threshold=pred_prob)
                    df = df.loc[(df["NC"]==0)&(df["type_pre"].isin([2,3]))]  # select only "leaf" & "axon and dendrite"
                    y_test = df["type_pre"].values
                    y_pred = df["type_post"].values
                    '''
                    # the table for correct/wrong nodes
                    _df = df.copy()
                    # _df = _df.rename(columns={"type_pre":"true", "type_post":"pred"})
                    # _df["correct"] = np.where(_df["true"]==_df["pred"], 1, 0)
                    # _df = _df[["nrn", "ID", "true", "pred", "correct"]]
                    # _df.to_csv(desktop + "predict_xgb_20200124_soma.csv", index=False)
                    _df.to_csv(desktop+"result.csv", index=False)
                    sys.exit()
                    '''
                else:
                    df = df.loc[df["type_pre"].isin([2, 3])]  # select all node & "axon and dendrite"
                    y_test = df["type_pre"].values
                    y_pred = df["prob"].values
                    y_pred = np.where(y_pred > pred_prob, 2, 3)
                '''
                else:
                    df = non_relabel(df)
                    df = df.loc[(df["NC"] == 0) & (df["type_pre"].isin([2, 3]))]  # select only "leaf" & "axon and dendrite"
                    y_test = df["type_pre"].values
                    y_pred = df["type_non_relabel"].values
                '''

                # evaluate report
                report = classification_report(y_test, y_pred, output_dict=True)
                acc_lst.append(report["accuracy"])
                # axon
                prc_lst.append(report["2"]["precision"])
                rcl_lst.append(report["2"]["recall"])
                f1_lst.append(report["2"]["f1-score"])
                # dendrite
                prc_lst0.append(report["3"]["precision"])
                rcl_lst0.append(report["3"]["recall"])
                f1_lst0.append(report["3"]["f1-score"])


                cm = confusion_matrix(y_test, y_pred, labels=[2,3])
                cm_lst.append(cm)

            df_eval["accuracy"] = acc_lst
            df_eval["precision"] = prc_lst
            df_eval["recall"] = rcl_lst
            df_eval["f1-score"] = f1_lst

            df_eval0["accuracy"] = acc_lst
            df_eval0["precision"] = prc_lst0
            df_eval0["recall"] = rcl_lst0
            df_eval0["f1-score"] = f1_lst0

            h_acc = acc_lst.index(max(acc_lst))
            h_thr = self.prob_thresholds[h_acc]
            df_pr.set_value("axon", "prc", df_eval.iloc[h_acc, 1])
            df_pr.set_value("axon", "rcl", df_eval.iloc[h_acc, 2])
            df_pr.set_value("dend", "prc", df_eval0.iloc[h_acc, 1])
            df_pr.set_value("dend", "rcl", df_eval0.iloc[h_acc, 2])

            self.axon_classified_dict[m]["confusion_matrix"] = cm_lst[0]
            self.axon_classified_dict[m]["confusion_matrix_highAcc"] = cm_lst[h_acc]
            self.axon_classified_dict[m]["evaluation_df1"] = df_eval
            self.axon_classified_dict[m]["evaluation_df0"] = df_eval0
            self.axon_classified_dict[m]["threshold_highAcc"] = h_thr
            self.axon_classified_dict[m]["prcRcl_highAcc"] = df_pr

            # save
            if self.post_relabel:
                with open(self.fname_dict[m], "wb") as file:
                    pickle.dump(self.axon_classified_dict[m], file=file)



        time.sleep(0.01)

        return


    def _show_evaluation(self):
        print_info_dict(self.axon_classified_dict["info_dict"])
        print("===========================================================================================")

        for key, v0 in self.axon_classified_dict.items():
            if key != "info_dict":
                print("Model =", key)

                acc = round(self.axon_classified_dict[key]["evaluation_df1"]["accuracy"].iloc[0], 3)
                print("confusion_matrix (accuracy",str(acc),"): (row: true, col: pred)\n", turn_confusingMatrix_to_DF(self.axon_classified_dict[key]["confusion_matrix"], self.post_relabel), "\n")
                if not self.post_relabel:
                    print(self.axon_classified_dict[key]["prcRcl_highAcc"], "\n")
                if self.post_relabel:
                    max_acc = round(max(self.axon_classified_dict[key]["evaluation_df1"]["accuracy"]), 3)
                    print("confusion_matrix (highest accuracy",str(max_acc),"at thr", round(self.axon_classified_dict[key]["threshold_highAcc"],3), "):\n",
                          turn_confusingMatrix_to_DF(self.axon_classified_dict[key]["confusion_matrix_highAcc"]), "\n")
                    print(self.axon_classified_dict[key]["prcRcl_highAcc"], "\n")

                print("evaluation_df1 (axon):\n", self.axon_classified_dict[key]["evaluation_df1"], "\n")
                print("evaluation_df0 (dendrite):\n", self.axon_classified_dict[key]["evaluation_df0"], "\n")
                print("===========================================================================================")

        return


    def _plot_result(self):
        print("Plotting...")

        df0 = None
        bar = progressbar.ProgressBar()
        time.sleep(0.01)

        # 1.
        for m in bar(list(self.models.keys())):
            # Full "original" level tree
            if df0 is None:

                ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level)
                ax0 = ax0.load_data(["forTrain"])
                df0 = ax0.prepare_dict["forTrain"]
                del ax0
                gc.collect()

                # with open("./data/nrn_cleaned/prepare_all", "rb") as file:
                #     df0 = pickle.load(file)

                # for nrn in self.test_nrn_lst:
                #     n1 = Data_Cleaner(self.input_folder, nrn, self.remove_method, self.target_level)
                #     n1 = n1.load_data()
                #     _df = update_parent_col(n1.df, n1.df_dis, n1.tree_node_dict, self.child_col, self.parent_col)
                #     if df0 is None:
                #         df0 = _df
                #     else:
                #         df0 = df0.append(_df)
                #     del _df, n1
                #     gc.collect()
                # df0 = df0.sort_values(['nrn', 'ID']).reset_index(drop=True)
                # df0 = df0.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'type_pre']]
            else:
                pass


            df = self.axon_classified_dict[m]["avg_result"]
            df = pd.merge(df0, df, how="left", on=['nrn', 'ID'])
            # df = df.loc[~df["nrn"].isin(group_dict["Type_LOB_OG"])]
            # df = df.loc[df["nrn"].isin(neuron_dict["lowest_prc_lst"])]

            # post relabel
            df = post_relabel(df, threshold=self.prob_threshold)

            # plot result level tree
            lst0 = []
            lst1 = []
            # b_lst = ['Cha-F-000031', 'Cha-F-000050', 'Cha-F-100041', 'Cha-F-100065', 'Cha-F-100092', 'Cha-F-100117', 'Cha-F-200013', 'Cha-F-200068', 'Cha-F-300072', 'Cha-F-300129', 'Cha-F-300152', 'Cha-F-300160', 'Cha-F-400012', 'Cha-F-400017', 'Cha-F-400260', 'Cha-F-500028', 'Cha-F-500046', 'Cha-F-500120', 'Cha-F-600158', 'Cha-F-800020', 'Gad1-F-000056', 'Gad1-F-000066', 'Gad1-F-000157', 'Gad1-F-000167', 'Gad1-F-100004', 'Gad1-F-200375', 'Gad1-F-200389', 'Gad1-F-300027', 'Gad1-F-300099', 'Gad1-F-300104', 'Gad1-F-300121', 'Gad1-F-300123', 'Gad1-F-400017', 'Gad1-F-400295', 'Gad1-F-400385', 'Gad1-F-400400', 'Gad1-F-500035', 'Gad1-F-500071', 'Gad1-F-500220', 'Gad1-F-500661', 'Gad1-F-600077', 'Gad1-F-600081', 'Gad1-F-600084', 'Gad1-F-700275', 'Gad1-F-800013', 'Gad1-F-800046', 'Gad1-F-800276', 'Gad1-F-800392', 'Gad1-F-900119', 'TH-F-000048', 'Tdc2-F-200009', 'Tdc2-F-300003', 'Tdc2-F-300014', 'Tdc2-F-300036', 'Tdc2-F-300042', 'Tdc2-F-400002', 'Tdc2-F-400009', 'Tdc2-F-400026', 'Tdc2-F-600000', 'Trh-F-500059', 'VGlut-F-000259', 'VGlut-F-000485', 'VGlut-F-200574', 'VGlut-F-300243', 'VGlut-F-300517', 'VGlut-F-300584', 'VGlut-F-300596', 'VGlut-F-500626', 'VGlut-F-500853', 'VGlut-F-600152', 'VGlut-F-600243', 'VGlut-F-600296', 'VGlut-F-600644', 'VGlut-F-600667', 'VGlut-F-600751', 'VGlut-F-700072', 'VGlut-F-700163', 'VGlut-F-700230', 'VGlut-F-700402', 'VGlut-F-800076', 'VGlut-F-800161', 'VGlut-F-800224', 'VGlut-F-800284', 'VGlut-F-800305', 'VGlut-F-900011', 'fru-F-400057', 'fru-F-400209', 'fru-F-400366', 'fru-F-500119', 'fru-F-500486', 'fru-F-500578', 'fru-F-800052', 'fru-F-900027', 'fru-F-900039', 'fru-M-400387', 'npf-F-200003', 'npf-F-200008', 'npf-F-200018']


            for nrn in df.nrn.unique():
            # for nrn in b_lst:
                # a = df.loc[(df["nrn"]==nrn)&(df["NC"]==0)&(df["type_pre"]!=0), "type_pre"].tolist()
                # b = df.loc[(df["nrn"]==nrn)&(df["NC"]==0)&(df["type_pre"]!=0), "type_post"].tolist()
                # if a == b:
                #     lst0.append(nrn)
                # else:
                #     lst1.append(nrn)
                ### fname
                save_path = input_folder + "nrn_plot/result_tree/" + m + "/"
                # fname = save_path + nrn + ".gv." + self.file_type

                plot_tree_typePost(df.loc[df["nrn"]==nrn], self.child_col, self.parent_col, self.type_col, "type_post", save_path, self.file_type, self.file_size, self.only_terminal, self.branch_col, self.show_node_id)
            # print(len(lst0))
            # print(lst0)
            # print(len(lst1))
            # print(lst1)
        return


    def correct_distribution(self, model, prob_threshold):
        '''
        Plot the correct distribution of the prediction of the specific brain area.
        :param model: str. Use 'xgb', 'svm', or 'dnn', etc.
        :param prob_threshold: float. Threshold for post_relabel().
        :return: save plots of correct distribution on the desktop.
        '''
        nrn_types = ["new_all_2", 'Type_AL_1', 'PFN', 'Type_MED_VMP']
        # nrn_types = ['PFN']

        for nrn_type in nrn_types:

            # load in classify result
            self._load_data()

            # load in full "original" level tree
            ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level)
            ax0 = ax0.load_data(["forEvaluate"])
            df0 = ax0.prepare_dict["forEvaluate"]
            del ax0
            gc.collect()

            # choose subsets
            df = self.axon_classified_dict[model]["avg_result"]
            df = pd.merge(df0, df, how="left", on=['nrn', 'ID'])
            if nrn_type == "new_all_2":
                pass
            else:
                df = df.loc[df["nrn"].isin(group_dict[nrn_type])]

            # post_relabel
            df = post_relabel(df, threshold=prob_threshold)
            df = df.loc[(df["NC"] == 0) & (df["type_pre"].isin([2, 3]))]  # select only "leaf" & "axon and dendrite"

            dict0 = {"acc":[], "rcl1":[], "rcl0":[]}
            _lst = df.nrn.unique().tolist()
            for i in _lst:
                df1 = df.loc[df["nrn"]==i].reset_index(drop=True)
                df1["correct"] = np.where(df1["type_pre"]==df1["type_post"], 1, 0)
                y_test = df1["type_pre"].values
                y_pred = df1["type_post"].values

                report = classification_report(y_test, y_pred, output_dict=True)
                dict0["acc"].append(report["accuracy"]*100)
                # axon
                dict0["rcl1"].append(report["2"]["recall"]*100)
                # dendrite
                dict0["rcl0"].append(report["3"]["recall"]*100)


            kde = False
            bins = int(10)  # 20
            hist_kws = {"range": [0, 100]}

            # plot acc distribution
            fig, ax = plt.subplots()
            x = np.array(dict0["acc"])
            sns.distplot(x, ax=ax, kde=kde, label='total', bins=bins, hist_kws=hist_kws)
            plt.legend()
            plt.title(nrn_type+"(nrn_"+str(len(_lst))+")", fontsize=16)
            plt.ylabel('number of nrn', fontsize=14)
            plt.xlabel('accuracy', fontsize=14)
            ax.set_ylim(0, len(x))
            plt.savefig(desktop + nrn_type +"_acc_"+str(len(x))+".pdf")
            plt.close()

            # plot prc distribution
            fig, ax = plt.subplots()
            sns.distplot(np.array(dict0["rcl1"]), ax=ax, kde=kde, label='axon', bins=bins, hist_kws=hist_kws, color="g")
            sns.distplot(np.array(dict0["rcl0"]), ax=ax, kde=kde, label="dendrite", bins=bins, hist_kws=hist_kws, color="r")
            plt.legend()
            plt.title(nrn_type+"(nrn_"+str(len(_lst))+")", fontsize=16)
            plt.ylabel('number of nrn', fontsize=14)
            plt.xlabel('recall', fontsize=14)
            ax.set_ylim(0, len(x))
            plt.savefig(desktop + nrn_type +"_rcl_"+str(len(x))+".pdf")
            plt.close()

        return


    def analyze_feature(self, model, top_ftrs, prob_threshold=None):
        nrn_types = ["new_all_2", "highest_prc_lst", "lowest_prc_lst"]
        # nrn_types = ["Type_LOB_OG", "not_Type_LOB_OG"]
        kde = False
        bins = int(100 / 5)
        # prob_threshold = 0.75


        # load in classify result
        self._load_data()
        df_info0 = self.axon_classified_dict[model]["avg_result"]

        # load in full "original" level tree
        ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level)
        ax0 = ax0.load_data(["forTrain"])
        df0 = ax0.prepare_dict["forTrain"]
        del ax0
        gc.collect()
        df_info1 = df0[['nrn', 'ID', 'type_pre']]
        df0 = pyramid(df0, self.features, self.pyr_layer, self.th_layer, for_train=True, empty_number=-1)
        df0 = pd.merge(df0, df_info0, how="left", on=['nrn', 'ID'])
        df0 = pd.merge(df0, df_info1, how="left", on=['nrn', 'ID'])
        df0 = df0.loc[df0["type_pre"].isin([2, 3])].reset_index(drop=True)

        for nrn_type in nrn_types:
            # choose subsets
            df = df0.loc[df0["nrn"].isin(neuron_dict[nrn_type])].reset_index(drop=True)

            # df0 = df0.loc[df0["nrn"].isin(neuron_dict["lowest_prc_lst"])]    # 0~10%
            # if nrn_type == "Type_LOB_OG":
            #     df = df0.loc[df0["nrn"].isin(group_dict[nrn_type])].reset_index(drop=True)
            # elif nrn_type == "not_Type_LOB_OG":
            #     df = df0.loc[~df0["nrn"].isin(group_dict["Type_LOB_OG"])].reset_index(drop=True)


            for i in range(top_ftrs):
                ftr_rnk = self.axon_classified_dict[model]["ftr_ranking"]
                ftr = ftr_rnk.ix[i, 0]
                if prob_threshold is None:
                    x1 = df.loc[df["type_pre"] == 2, ftr].to_list()
                    x0 = df.loc[df["type_pre"] == 3, ftr].to_list()
                else:
                    x1 = df.loc[(df["type_pre"]==2)&(df["prob"]>=prob_threshold), ftr].to_list()
                    x0 = df.loc[(df["type_pre"]==3)&(df["prob"]<=(1-prob_threshold)), ftr].to_list()
                r = [max(x1), max(x0), min(x1), min(x0)]
                hist_kws = {"range": [min(r), max(r)]}

                # plot prc distribution
                rnk = "(rnk_"+str(i)+")"
                if prob_threshold is None:
                    name1 = "".join([nrn_type, rnk])
                else:
                    thr = "(thr_"+str(prob_threshold)+")"
                    name1 = "".join([nrn_type, rnk, thr])
                sns.set(color_codes=True)
                sns.distplot(np.array(x1), kde=kde, label='axon', bins=bins, hist_kws=hist_kws)
                sns.distplot(np.array(x0), kde=kde, label="dendrite", bins=bins, hist_kws=hist_kws)
                plt.legend()
                plt.title(name1 + "(nodes_" + str(len(x1+x0)) + ")", fontsize=16)
                plt.ylabel('number of nodes', fontsize=14)
                plt.xlabel(ftr, fontsize=14)
                plt.savefig(desktop + name1 + ".pdf")
                plt.close()

            del df

        return


    def check_feature(self):
        # train
        ax0 = Prepare_Axon(self.input_folder, self.train_nrn_type, self.remove_method, self.target_level, self.augment_number)
        ax0 = ax0.load_data(["forTrain", "forTrainFk"])
        df = ax0.prepare_dict["forTrain"]
        fk = ax0.prepare_dict["forTrainFk"]
        del ax0

        for f in self.features:
            _df = df.loc[df[f]<0]
            _fk = fk.loc[fk[f]<0]
            if len(_df)>0:
                print("feature:", f)
                print("df: \n", _df)
            if len(_fk)>0:
                print("feature:", f)
                print("fk: \n", _fk)

        return





if __name__ == '__main__':
    ax0 = Classify_Axon(input_folder,
                        train_nrn_type,
                        test_nrn_type,
                        remove_method,
                        target_level,
                        models,
                        mdl_ensemble,
                        features,
                        pyramid_layer,
                        threshold_layer,
                        sample,
                        sample_pct,
                        sample_with_replacement,
                        augment_nrn_type,
                        augment_number,
                        overwrite=overwrite)

    ax0 = ax0.classify_data(rerun_mdl)

    ax0 = ax0.predict(prob_threshold=0.7)

    ax0.evaluate(relabel=relabel)

    # ax0.evaluate(relabel=relabel, prob_thresholds = [0.7])

    ax0.plot_result(prob_threshold=0.7, file_type='jpg', only_terminal=only_terminal, branch_col=branch_col, show_node_id=show_node_id)

    # ax0.correct_distribution(model="xgb", prob_threshold=0.75)

    # ax0.analyze_feature(model="xgb", top_ftrs=3)

    # ax0.check_feature()




########################################################################################################################
