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


########################################################################################################################
# Set up
########################################################################################################################
input_folder = '/data/'
nrn_lst = get_fileNames_from_directory(input_folder+"nrn_original/", file_type='swc', drop_file_type=True)
# nrn_lst = ['5HT1A-F-100004', '5HT1A-F-100032']
remove_method = 'leaf'
target_level = 5
# remove_method = None
# target_level = None
overwrite = False


########################################################################################################################
# Main Code
########################################################################################################################
class Data_Cleaner():
    def __init__(self,
                 input_folder,
                 nrn_name,
                 remove_method=None,
                 target_level=None,
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='type_pre',   # Here use pre_relabel type 'type_pre'; the original type is 'T'.
                 overwrite=False
                ):

        self.sys_dir = os.path.dirname(__file__)
        if all([remove_method is None, target_level is None]):
            self.fname = self.sys_dir + input_folder+"nrn_cleaned/"+nrn_name+".pkl"
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method+str(target_level)
            self.fname = self.sys_dir + input_folder+"nrn_cleaned/"+"_".join([nrn_name, _methodLevel])+".pkl"
        else:
            sys.exit("\n remove_method = None or str; target_level = None or int! Check Prepare_Axon.")

        self.input_folder = input_folder
        self.nrn_name = nrn_name
        self.remove_method = remove_method
        self.target_level = target_level
        self.origin_cols = origin_cols
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        self.distance_decimal = None
        self.remove_iterate = 20

        self.df_axon = None
        self.df_mix = None
        self.axon_feature = None
        self.mix_feature = None

        return


    def load_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._load_swc()
            self._remove_small_leaf()
            self._save_data()
            self._load_data()
        return self


    def _is_ready(self):
        if(os.path.exists(self.fname)):
            return True
        else:
            return False


    def _load_data(self):
        with open(self.fname, "rb") as file:
            self.df, self.df_dis, self.df_axon, self.df_mix, self.df_level_descend, self.tree_node_dict, self.polarity_dict, self.axon_feature, self.mix_feature = pickle.load(file)
        return


    def _load_swc(self):
        path = os.path.dirname(__file__)
        fname_swc = path + self.input_folder + "nrn_original/" + self.nrn_name + ".swc"
        nrn = nm.io.swc.read(fname_swc)
        self._df0 = pd.DataFrame(nrn.data_block, columns=self.origin_cols)
        self._df0[['T', 'ID', 'PARENT_ID']] = self._df0[['T', 'ID', 'PARENT_ID']].astype('int')
        self._df0['T'] = self._df0['T'].abs()
        return self._df0

    def _remove_small_leaf(self):
        # Remove small leaf and Create new df
        level_lst = []
        threshold_lst = []
        df = self._df0
        for i in range(self.remove_iterate + 1):
            # 1. Remove small leaf
            if i == 0:
                # skip the first time removal to create level tree
                threshold = 0
                threshold_lst.append(threshold)
                pass
            else:
                # Find the distance threshold for remove small leaf
                threshold = find_distanceX_of_maxNumY(df_dis, tree_node_dict, self.remove_method, self.distance_decimal,
                                                      view=False, save_path=None)
                threshold_lst.append(threshold)
                # Remove small leaf
                df = remove_small_leaf(df, df_dis, tree_node_dict, self.origin_cols, less_than=threshold)

            # 2. Created Cols: nrn_name, NC, Q, type_pre & DFs: df_dis
            # nrn_name
            df["nrn"] = self.nrn_name

            # Create child number col and dictionary of leaf/fork/root lists
            df, tree_node_dict = neuron_childNumCol(df, self.child_col, self.parent_col)

            # Create ancestors and path dfs
            df_anc, df_path = neuron_ancestors_and_path(df, self.child_col, self.parent_col)

            # Create branches (level tuple list)
            branch_lst = neuron_level_branch(df_path, tree_node_dict)

            # Count the level for each point
            df, max_level, first_fork = neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst, self.child_col,self.parent_col)
            level_lst.append(max_level)

            # Create distances of branches and create Q col
            df, df_dis, dis_lst = neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst,first_fork, self.distance_decimal, self.child_col, self.parent_col)
            # print(df_dis)

            # Create type_pre col
            # update parent col, select fork and leaf nodes
            _df = update_parent_col(df, df_dis, tree_node_dict, self.child_col, self.parent_col, ["fork", "leaf"])
            # create type_pre col
            _df = pre_relabel(_df)
            # merge type_pre to df
            _df = _df.loc[:, [self.child_col, self.type_col]]
            df = pd.merge(df, _df, how="left", on=self.child_col)
            df[self.type_col] = np.where(np.isnan(df[self.type_col]), 0, df[self.type_col])
            df[[self.type_col]] = df[[self.type_col]].astype('int')
            del _df



            # 3. Level tree data
            if all([not os.path.exists(self.input_folder+self.nrn_name+".pkl"), i == 0]):
                # Create df(main)
                self.df_0 = df.sort_values(["dp_level", self.child_col], ascending=[False, True])

                # df_dis(distance) and tree_node_dict
                self.df_dis_0 = df_dis
                self.tree_node_dict_0 = tree_node_dict

                # Create polarity_dict
                self.polarity_dict_0 = neuron_polarity_dict(self.df_0, self.child_col, self.type_col)

                self.df_level_descend_0 = None


            # 4. Reduce tree data
            # Stop the loop
            if all([self.remove_method is not None, self.target_level is not None]):
                if any([max_level <= self.target_level]):
                    break


                elif all([len(level_lst) >= 3, len(threshold_lst) >= 3]):
                    if all([level_lst[-1] == level_lst[-3], threshold_lst[-1] == threshold_lst[-3]]):
                        break

            else:
                break


        # 5. Reduce tree data
        # Create df(main)
        self.df_1 = df.sort_values(["dp_level", self.child_col], ascending=[False, True])

        # df_dis (distance), df_level_descend, tree_node_dict
        self.df_dis_1 = df_dis
        self.df_level_descend_1 = pd.DataFrame(OrderedDict({'level': level_lst, 'threshold': threshold_lst}))
        self.tree_node_dict_1 = tree_node_dict

        # Create polarity_dict
        self.polarity_dict_1 = neuron_polarity_dict(self.df_1, self.child_col, self.type_col)

        return


    def _save_data(self):
        if all([self.remove_method is None, self.target_level is None]):
            with open(self.fname, "wb") as file:
                pickle.dump([self.df_0, self.df_dis_0, self.df_axon, self.df_mix, self.df_level_descend_0, self.tree_node_dict_0, self.polarity_dict_0, self.axon_feature, self.mix_feature], file=file)

        elif any([overwrite, not os.path.exists(self.input_folder+"nrn_cleaned/"+self.nrn_name+".pkl")]):
            path = self.sys_dir + self.input_folder + "nrn_cleaned/"+self.nrn_name+".pkl"
            with open(path, "wb") as file:
                pickle.dump([self.df_0, self.df_dis_0, self.df_axon, self.df_mix, self.df_level_descend_0, self.tree_node_dict_0, self.polarity_dict_0, self.axon_feature, self.mix_feature], file=file)
            with open(self.fname, "wb") as file:
                pickle.dump([self.df_1, self.df_dis_1, self.df_axon, self.df_mix, self.df_level_descend_1, self.tree_node_dict_1, self.polarity_dict_1, self.axon_feature, self.mix_feature], file=file)

        else:
            with open(self.fname, "wb") as file:
                pickle.dump([self.df_1, self.df_dis_1, self.df_axon, self.df_mix, self.df_level_descend_1, self.tree_node_dict_1, self.polarity_dict_1, self.axon_feature, self.mix_feature], file=file)
        return



if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for nrn in bar(nrn_lst):
        # nrn0 = Data_Cleaner(input_folder, nrn, overwrite=overwrite)
        nrn1 = Data_Cleaner(input_folder, nrn, remove_method, target_level, overwrite=overwrite)

        # n0 = nrn0.load_data()
        n1 = nrn1.load_data()

    time.sleep(0.01)



########################################################################################################################
