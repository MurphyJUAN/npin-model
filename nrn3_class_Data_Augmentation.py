# Purpose:
# 1. Create original dfs: original data and distance
# 2. Create reduced dfs: reduced data, distance, and level_descend
# 3. Output:
#       5-HT1B-F-500013.pkl: original data of a nrn.
#       5-HT1B-F-500013_leaf5.pkl: use leaf method to remove and the target level is 5.


########################################################################################################################
from .util import *
from .nrn_params import *
from .settings import *
from .nrn3_class_Data_Cleaner import Data_Cleaner
from .nrn3_class_Create_Axon_Feature import Create_Axon_Feature


########################################################################################################################
# Set up
########################################################################################################################
input_folder = './data/'
nrn_lst = neuron_dict["new_all_2"]
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
augment = 1
overwrite = False


########################################################################################################################
# Main Code
########################################################################################################################
class Data_Augmentation():
    def __init__(self,
                 input_folder,
                 nrn_name,
                 remove_method=None,
                 target_level=None,
                 augment=10,
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T',
                 overwrite=False
                ):


        self.input_folder = input_folder
        self.nrn_name = nrn_name
        self.remove_method = remove_method
        self.target_level = target_level
        self.augment = augment
        self.origin_cols = origin_cols
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        self.distance_decimal = 0
        self.remove_iterate = 20
        self.vibrate_amplitude = 1/3   # this will time "the min length" of the branch of a neuron

        self.df_axon = None
        self.df_mix = None
        self.axon_feature = None
        self.mix_feature = None


        if self.augment < 1:
            sys.exit("\n augment should >= 1! Please check Data_Augmentation().")

        # nrn_name for fake data
        self.fk_name_lst = ["fk" + str(i) + "_" + self.nrn_name for i in range(self.augment)]

        # fname for fake data
        self.fname_dict = {}
        for x in self.fk_name_lst:
            if all([remove_method is None, target_level is None]):
                _fname0 = x
            elif all([type(remove_method) is str, type(target_level) is int]):
                _methodLevel = remove_method + str(target_level)
                _fname0 = "_".join([x, _methodLevel])
            else:
                sys.exit("\n remove_method = None or str; target_level = None or int! Check Prepare_Axon.")

            self.fname_dict[x] = input_folder+"nrn_cleaned/"+_fname0+".pkl"


        return



    def generate_data(self):
        if all([remove_method is None, target_level is None]):
            self._generate_level_tree()
        elif all([type(remove_method) is str, type(target_level) is int]):
            self._generate_reduce_tree()
        return self


    def create_axon_featrue(self):
        self._create_axon_featrue()
        return self




    def _generate_level_tree(self):
        # print("Run Data_Augmentation(): generate data...")
        n1 = Data_Cleaner(self.input_folder, self.nrn_name)
        n1 = n1.load_data()
        df = n1.df
        df_dis = n1.df_dis
        tnd = n1.tree_node_dict
        df_axon = df_mix = axon_feature = mix_feature = None


        for fk_nrn_name, fname in self.fname_dict.items():
            if (os.path.exists(fname)):
                continue

            df_0 = df.copy()
            df_dis_0 = df_dis.copy()
            df_0["nrn"] = fk_nrn_name
            df_dis_0["direct_dis_des_anc"] = 0

            nodes0 = dict_merge_value(tnd)
            nodes = partition(nodes0, 3, shuffle_list=True)

            vibrate = ["x", "y", "z"]

            r0 = min(df_dis["len"])*self.vibrate_amplitude
            r = [r0, -r0]

            # A.
            # 1. Vibration
            for i in range(3):
                _n = nodes[i]
                _v = random.choice(vibrate)
                _r = random.choice(r)

                df_0[_v] = np.where(df_0[self.child_col].isin(_n), df_0[_v]+_r, df_0[_v])

            # 2. Calculate the new distance
            for _idx in range(len(df_dis_0)):
                # Calculate distance
                _s = df_0.loc[df_0[self.child_col]==tnd["root"][0], ['x', 'y', 'z']]
                _s = [tuple(x) for x in _s.values]

                _d0 = df_dis_0.ix[_idx, "descendant"]
                _d = df_0.loc[df_0[self.child_col]==_d0, ['x', 'y', 'z']]
                _d = [tuple(x) for x in _d.values]

                _a0 = df_dis_0.ix[_idx, "ancestor"]
                _a = df_0.loc[df_0[self.child_col] == _a0, ['x', 'y', 'z']]
                _a = [tuple(x) for x in _a.values]

                tuples_ds = _s + _d
                _ds = calculate_distance(tuples_ds)
                df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"]==_d0, _ds, df_dis_0["direct_dis_des_soma"])

                tuples_dp = _a + _d
                _dp = calculate_distance(tuples_dp)
                df_dis_0.ix[_idx, "direct_dis_des_anc"] = _dp

            # 3. Find out nodes which violate the rule
            _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)


            # B.
            # Adjust those nodes which violate the rule
            while len(_df_dis) > 0:

                # 1. Recover x,y,z from vibration
                for _idx in range(len(_df_dis)):
                    _d0 = _df_dis.ix[_idx, "descendant"]
                    _a0 = _df_dis.ix[_idx, "ancestor"]

                    for _n0 in [_d0, _a0]:
                        _n = df.loc[df[self.child_col] == _n0, ['x', 'y', 'z']].values
                        _row = df_0.index[df[self.child_col] == _n0].tolist()[0]
                        df_0.ix[_row, "x"] = _n[0, 0]
                        df_0.ix[_row, "y"] = _n[0, 1]
                        df_0.ix[_row, "z"] = _n[0, 2]

                # 2. Calculate the new distance
                for _idx in range(len(df_dis_0)):
                    # Calculate distance
                    _s = df_0.loc[df_0[self.child_col] == tnd["root"][0], ['x', 'y', 'z']]
                    _s = [tuple(x) for x in _s.values]

                    _d0 = df_dis_0.ix[_idx, "descendant"]
                    _d = df_0.loc[df_0[self.child_col] == _d0, ['x', 'y', 'z']]
                    _d = [tuple(x) for x in _d.values]

                    _a0 = df_dis_0.ix[_idx, "ancestor"]
                    _a = df_0.loc[df_0[self.child_col] == _a0, ['x', 'y', 'z']]
                    _a = [tuple(x) for x in _a.values]

                    tuples_ds = _s + _d
                    _ds = calculate_distance(tuples_ds)
                    df_dis_0["direct_dis_des_soma"] = np.where(df_dis_0["descendant"] == _d0, _ds,
                                                               df_dis_0["direct_dis_des_soma"])

                    tuples_dp = _a + _d
                    _dp = calculate_distance(tuples_dp)
                    df_dis_0.ix[_idx, "direct_dis_des_anc"] = _dp

                # 3. Find out nodes which violate the rule
                _df_dis = df_dis_0.loc[df_dis_0["len"] < df_dis_0["direct_dis_des_anc"]].reset_index(drop=True)

            # _lst = df_dis_0.loc[df_dis_0["len"] > df_dis_0["direct_dis_des_anc"], "descendant"].tolist()
            # print(len(_lst))

            _df_dis = _df_dis.drop(["direct_dis_des_anc"], 1)


            # save data
            with open(fname, "wb") as file:
                    pickle.dump([df_0, df_dis_0, df_axon, df_mix, n1.df_level_descend, n1.tree_node_dict, n1.polarity_dict, axon_feature, mix_feature], file=file)

        return



    def _generate_reduce_tree(self):
        # print("Run Data_Augmentation(): generate data...")
        n1 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
        n1 = n1.load_data()
        df = n1.df
        df_dis = n1.df_dis
        df_axon = df_mix = axon_feature = mix_feature = None

        # drop columns
        df = df.drop(['x', 'y', 'z'], axis=1)
        df_dis = df_dis.drop(['direct_dis_des_soma'], axis=1)


        for fk_name, fname in self.fname_dict.items():
            if (os.path.exists(fname)):
                continue

            # copy data from Data_Cleaner
            df_0 = df.copy()
            df_dis_0 = df_dis.copy()

            # load fake level tree
            _fname = input_folder+"nrn_cleaned/"+fk_name+".pkl"
            if not (os.path.exists(_fname)):
                _n = Data_Augmentation(self.input_folder, self.nrn_name, None, None, self.augment)
                _n._generate_level_tree()
            with open(_fname, "rb") as file:
                df_l, df_dis_l, _, _, _, _, _, _, _ = pickle.load(file)

            # create final column list
            df_cols = list(df_l)
            df_dis_cols = list(df_dis_l)

            # select wanted column
            df_l = df_l[['x', 'y', 'z', self.child_col]]
            df_dis_l = df_dis_l[['descendant', 'direct_dis_des_soma', 'direct_dis_des_anc']]

            # merge DFs
            df_0 = pd.merge(df_0, df_l, on=[self.child_col], how="left")
            df_dis_0 = pd.merge(df_dis_0, df_dis_l, on=["descendant"], how="left")

            # rearrange columns
            df_0 = df_0[df_cols]
            df_dis_0 = df_dis_0[df_dis_cols]

            # save data
            with open(fname, "wb") as file:
                    pickle.dump([df_0, df_dis_0, df_axon, df_mix, n1.df_level_descend, n1.tree_node_dict, n1.polarity_dict, axon_feature, mix_feature], file=file)


        return



    def _create_axon_featrue(self):
        # Make sure leaf5 reduce tree exist to run shape_feature() in Create_Axon_Feature().
        _n = Data_Augmentation(self.input_folder, self.nrn_name, "leaf", 5, self.augment)
        _n._generate_reduce_tree()
        # Create axon feature
        for nrn in self.fk_name_lst:
            ax0 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level, overwrite=self.overwrite)
            ax0 = ax0.load_data()



        return


    def _rename_nrn(self):

        for fk_nrn_name, fname in self.fname_dict.items():
            if (os.path.exists(fname)):
                with open(fname, "rb") as file:
                    df, df_dis, df_axon, df_mix, df_level_descend, tree_node_dict, polarity_dict, axon_feature, mix_feature = pickle.load(file)

            df["nrn"] = fk_nrn_name
            # df_axon["nrn"] = fk_nrn_name


            # save data
            with open(fname, "wb") as file:
                    pickle.dump([df, df_dis, df_axon, df_mix, df_level_descend, tree_node_dict, polarity_dict, axon_feature, mix_feature], file=file)




        return




if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for nrn in bar(nrn_lst):
        n1 = Data_Augmentation(input_folder, nrn, remove_method, target_level, augment, overwrite=overwrite)

        n1 = n1.generate_data()

        n1 = n1.create_axon_featrue()

        # n1._rename_nrn()

    time.sleep(0.01)



########################################################################################################################
