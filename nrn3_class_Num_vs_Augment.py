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
from nrn3_class_Classify_Axon import Classify_Axon



########################################################################################################################
# Set up
########################################################################################################################
input_folder = './data/'
# data
train_nrn_type = ["new_all_2"]  # "all", "normal", "small", "multiple", or "axonNear"
test_nrn_type = ["new_all_2"]  # "all", "normal", "small", "multiple", or "axonNear"
remove_method = None    # None or "leaf"
target_level = None     # None or int

# model
models = ["xgb"]      # "svm", "gbdt", "rf", "ridge", "xgb", "dnn"
mdl_ensemble = 1                 # int >= 1
features = ['l', 'nl', 's', 'ns', 'ds', 'nds', 'ro', 'c', 'rc']    # 'l', 'nl', 's', 'ns', 'ds', 'nds', 'ro', 'c', 'rc'
pyramid_layer = 1       # >= 1, 1: only target node, 2: target node + 1 generation, 3: target node + 2 generation.
threshold_layer = 2     # >= 1, 1: need no descendants, 2: need at least 1 generation
sample = "random"   # "random" or "region"
# sample_pct_lst = [0.345, 0.452, 0.562, 0.672]   # random
sample_pct_lst = [0.452]   # random
# sample = "region"   # "random" or "region"
# sample_pct_lst = [0.655, 0.54, 0.43, 0.325]   # region
sample_with_replacement = 50    # int

# data augmentation
augment_nrn_type = ["new_all_2"]
# augment_number_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
augment_number_lst = [0, 10, 20, 50]

# other
pred_prob_lst = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# pred_prob_lst = [0.65, 0.7, 0.75]
overwrite = False


########################################################################################################################
# Main Code
########################################################################################################################
class Num_vs_Augment:

    def __init__(self,
                 input_folder,
                 train_nrn_type,
                 test_nrn_type,
                 remove_method=None,
                 target_level=None,
                 models=["xgb", "dnn"],
                 mdl_ensemble=2,
                 features=['l', 'nl', 's', 'ns', 'ds', 'nds', 'ro', 'c', 'rc'],
                 pyramid_layer=5,
                 threshold_layer=4,
                 sample="random",
                 sample_pct_lst=None,
                 sample_with_replacement=10,
                 augment_nrn_type=None,
                 augment_number_lst=[1,2,3,4,5,6,7,8,9,10],
                 pred_prob=0.8,
                 child_col='ID',
                 parent_col="PARENT_ID",
                 type_col='type_pre',   # Here use pre_relabel type 'type_pre'; the original type is 'T'.
                 overwrite=False,
                 seed=123
                 ):

        self.axon_classified_dict = {}
        self.classify_target = "axon"
        self.input_folder = input_folder
        self.train_nrn_type = sorted(train_nrn_type)
        self.test_nrn_type = sorted(test_nrn_type)
        self.mdl_ensemble = mdl_ensemble
        self.remove_method = remove_method
        self.target_level = target_level
        self.models = models
        self.features = sorted(features)
        self.pyr_layer = pyramid_layer
        self.th_layer = threshold_layer
        self.sample = sample
        self.with_replacement = sample_with_replacement
        self.augment_nrn_type = augment_nrn_type
        self.augment_number_lst = augment_number_lst
        self.seed = seed
        self.pred_prob = pred_prob
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        # Prepare_Axon
        self.all_nrn = ['new_all_2']
        self.max_aug_num = 5
        # Other configs
        self.branch = 2
        if sample_pct_lst is None:
            self.sample_pct_lst = [1/7]
        else:
            self.sample_pct_lst = sample_pct_lst



        return





    def run(self):
        s_dict = {"0.345":"150", "0.452":"125", "0.562":"100", "0.672":"75"}
        df = pd.DataFrame(index=self.augment_number_lst)
        xs = []
        ys = []
        zs = []


        augment_nrn_type = self.augment_nrn_type
        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for augment_number in bar(self.augment_number_lst):
            for sample_pct in self.sample_pct_lst:
                ax0 = Classify_Axon(self.input_folder,
                                    self.train_nrn_type,
                                    self.test_nrn_type,
                                    self.remove_method,
                                    self.target_level,
                                    self.models,
                                    self.mdl_ensemble,
                                    self.features,
                                    self.pyr_layer,
                                    self.th_layer,
                                    self.sample,
                                    sample_pct,
                                    self.with_replacement,
                                    augment_nrn_type,
                                    augment_number)
                ax0 = ax0.load_data()
                _df = ax0["xgb"]["evaluation_df1"].reset_index()
                _df["index"] = round(_df["index"], 2)
                val = _df.loc[_df["index"] == self.pred_prob, "accuracy"].values[0]

                xs.append(augment_number)
                ys.append(sample_pct)
                zs.append(val)
                df.at[augment_number, str(sample_pct)] = val

                del ax0, _df, val
                gc.collect()
        time.sleep(0.01)

        df = df.rename(columns=s_dict)

        print(self.pred_prob)
        print(df)

        # # Plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # For each set of style and range settings, plot n random points in the box
        # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        # # for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        # for m, zlow, zhigh in [('o', -50, -25)]:
        #     xs = np.asarray(xs)
        #     ys = np.asarray(ys)
        #     zs = np.asarray(zs)
        #     ax.scatter(xs, ys, zs, marker=m)
        #
        # ax.set_xlabel('Augment')
        # ax.set_ylabel('Num of nrn')
        # ax.set_zlabel('Accuracy')
        #
        # plt.show()

        return






if __name__ == '__main__':
    for pred_prob in pred_prob_lst:
        ax0 = Num_vs_Augment(input_folder,
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
                            sample_pct_lst,
                            sample_with_replacement,
                            augment_nrn_type,
                            augment_number_lst,
                            pred_prob,
                            overwrite=overwrite)

        ax0 = ax0.run()




########################################################################################################################
