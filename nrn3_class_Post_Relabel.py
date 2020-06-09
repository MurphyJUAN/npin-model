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
from nrn3_class_DNN import DNN
from nrn3_class_Data_Cleaner import Data_Cleaner
from nrn3_class_Create_Axon_Feature import Create_Axon_Feature



########################################################################################################################
# Set up
########################################################################################################################
input_folder = './data/'
# data
train_nrn_type = ["new_all_3"]  # "all", "normal", "small", "multiple", or "axonNear"
test_nrn_type = ["new_all_3"]  # "all", "normal", "small", "multiple", or "axonNear"
remove_method = None    # None or "leaf"
target_level = None     # None or int
pyramid_layer = 3       # int >= 1, 1: only target node, 2: target node + 1 generation, 3: target node + 2 generation.
threshold_layer = 1     # int >= 1, 1: need no descendants, 2: need at least 1 generation
sample = "region"   # "random" or "region"
sample_pct = 0.585    # (0, 1)
sample_with_replacement = 20    # int

# data augmentation
augment_nrn_type = ["new_all_3"]
augment_number = 0     # int >= 0

# models
model_dict = {"soma": ["xgb", ['s', 'ns', 'ds', 'nds'], 1],
              "local": ["xgb", ['l', 'nl', 'as1', 'c1', 'rc'], 1]
              }     # {"model_name": [model, features, mdl_ensemble]}

# other
overwrite = False    # True: overwrite the predict result of classify_data(), evaluation of evaluate()
relabel = True       # True: use post_relabel to evaluate

only_terminal=True   # True: plot terminal only
branch_col="l"       # None: show no branch length; "l": show branch length
show_node_id=True    # True: plot nodes with id number


########################################################################################################################
# Main Code
########################################################################################################################
class Post_Relabel:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 pyramid_layer=6,
                 threshold_layer=4,
                 sample="random",
                 sample_pct=None,
                 sample_with_replacement=10,
                 augment_nrn_type=None,
                 augment_number=0,
                 model_dict={"mdl1": ["xgb", ['l', 'nl', 'ro', 'c', 'rc'], 1], "mdl2": ["xgb", ['s', 'ns', 'ds', 'nds'], 1]},
                 child_col='ID',
                 parent_col="PARENT_ID",
                 type_col='type_pre',   # Here use pre_relabel type 'type_pre'; the original type is 'T'.
                 overwrite=False
                 ):

        self.axon_classified_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.pyr_layer = pyramid_layer
        self.th_layer = threshold_layer
        self.sample = sample
        self.sample_pct = sample_pct
        self.with_replacement = sample_with_replacement
        self.augment_nrn_type = augment_nrn_type
        self.augment_number = augment_number
        self.model_dict = model_dict
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite


        # Other configs
        self.branch = 2
        if sample_pct is None:
            self.sample_pct = 1/7
        else:
            self.sample_pct = sample_pct


        # fname
        for k, v in self.model_dict.items():
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
            self.features = sorted(v[1])
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
            self.mdl_ensemble = {}
            m = v[0]
            mdl_ensemble = v[2]
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

            v.append(input_folder + "nrn_result/" + "_".join([_fname0, _m]) + ".pkl")



        return




    def evaluate(self, relabel=True, show=True, prob_thresholds=np.arange(0.5, 1, 0.05)):
        self.post_relabel = relabel
        self.prob_thresholds = prob_thresholds
        if not self.post_relabel:
            self.prob_thresholds = [0.5]    # if not post relabel, pred threshold will fixed at 0.5
        if not self._is_ready():
            sys.exit("No classified result to be evaluated.")
        self._load_data()
        self._data_info()
        self._evaluate()
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
        for _, v in self.model_dict.items():
            if os.path.exists(v[3]):
                x = True
            else:
                x = False
                break
        return x


    def _load_data(self):
        self.axon_classified_dict = {}
        for k, v in self.model_dict.items():
            with open(v[3], "rb") as file:
                self.axon_classified_dict[k] = pickle.load(file)
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
            "augment_nrn_type": self.augment_nrn_type,
            "augment_number": self.augment_number,
            "post_relabel": self.post_relabel
        }

        return


    def _evaluate(self):
        print("Evaluate models...")

        print_info_dict(self.axon_classified_dict["info_dict"])
        print("===========================================================================================")

        ax0 = Prepare_Axon(self.input_folder, self.test_nrn_type, self.remove_method, self.target_level)
        ax0 = ax0.load_data(["forEvaluate"])
        df0 = ax0.prepare_dict["forEvaluate"]
        del ax0
        gc.collect()

        lst_of_df = []
        for k in self.model_dict.keys():
            df = self.axon_classified_dict[k]["avg_result"]
            df = pd.merge(df0, df, how="left", on=['nrn', 'ID'])
            lst_of_df.append([k, df])

        hyper_relabel(lst_of_df, threshold=0.8, reduce_th=0.05)

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




if __name__ == '__main__':
    ax0 = Post_Relabel(input_folder,
                        train_nrn_type,
                        test_nrn_type,
                        remove_method,
                        target_level,
                        pyramid_layer,
                        threshold_layer,
                        sample,
                        sample_pct,
                        sample_with_replacement,
                        augment_nrn_type,
                        augment_number,
                        model_dict,
                        overwrite=overwrite)



    ax0.evaluate(relabel=relabel)

    # ax0.plot_result(prob_threshold=0.75, file_type='jpg', only_terminal=only_terminal, branch_col=branch_col, show_node_id=show_node_id)




########################################################################################################################
