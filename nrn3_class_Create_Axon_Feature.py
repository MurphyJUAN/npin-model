# Purpose:
# 1. Create feature for each node (DO NOT drop first fork):
# csu:
# len
# len_des_soma
# direct_dis_des_soma
#
# ktc:
# dis
# soma dis
# original angle
# river layer
# simple layer
# deep layer from simple layer
# angle
# terminal number
# self-compared angle min
# self-compared angle max
# direct distance to soma
# the related length to the last node
# relative length to last node
# relative length to soma
# relative direct distance to soma
#
# 2. label={'axon':1, 'dendrite':0}
# 3. Output:
#       prepared_axon_leaf5.pkl: prepare nrns.


########################################################################################################################
from .util import *
from .nrn_params import *
from .settings import *
from .nrn3_class_Data_Cleaner import Data_Cleaner


########################################################################################################################
# Set up
########################################################################################################################
input_folder = '/data/'
nrn_lst = neuron_dict["new_all_2"]
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
overwrite = False


########################################################################################################################
# Main Code
########################################################################################################################
class Create_Axon_Feature:

    def __init__(self,
                 input_folder,
                 nrn_name,
                 remove_method=None,
                 target_level=None,
                 origin_cols=['nrn', 'ID', 'PARENT_ID', 'NC', 'dp_level', 'T', 'type_pre'],
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

        self.feature_info = {
            # distance to parent
            "l": "len",   # length
            "nl": "norm_len",
            # distance to soma
            "s": "len_des_soma",    # length
            "ns": "norm_len_des_soma",
            "ds": "direct_dis_des_soma",  # direct distance
            "nds": "norm_direct_dis_des_soma",
            # shape
            "ro": "ratio_ortho",    # along w/ the trunk
            "ro2": "ratio2_ortho",
            "c": "curvature_ave",  # average the curvature
            "as1": "ratio_com",    # new definition: center of mass
            "as2": "ratio2_com",
            "c1": "curvature",       # new definition: w/o average the curvature
            "cr": "curvature_r",
            "cs": "curvature_superposition",
            "v": "volume",
            "vl": "ave_volume_length",
            "vt": "ave_volume_terminals",
            "as": "aspect_ratio",
            # other
            "rc": "ratio_children",
            "s_ex": "s/600",
            "l_ex": "l/18",
            "ds_ex": "ds/300"

        }


        self.branch = 2


        return


    def load_data(self):
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
            if self._is_feature_ready():
                pass
            else:
                self._create_df_from_Data_Cleaner()
                self._save_data()
                self._load_data()
        else:
            self._create_df_from_Data_Cleaner()
            self._save_data()
            self._load_data()
        return self


    def _is_ready(self):
        if os.path.exists(self.fname):
            return True
        else:
            return False


    def _load_data(self):
        with open(self.fname, "rb") as file:
            self.df, self.df_dis, self.df_axon, self.df_mix, self.df_level_descend, self.tree_node_dict, self.polarity_dict, self.axon_feature, self.mix_feature = pickle.load(file)
        return


    def _is_feature_ready(self):
        if self.df_axon is None:
            return False
        else:
            col_lst = list(self.df_axon)
            ftr_lst = list(self.feature_info.keys())
            if all(elem in col_lst for elem in ftr_lst):
                return True
            else:
                return False


    def _create_df_from_Data_Cleaner(self):
        # Read in data
        n0 = Data_Cleaner(self.input_folder, self.nrn_name, self.remove_method, self.target_level)
        n0 = n0.load_data()
        # Remove branch > 2 and relabel L
        df, df_dis = remove_extra_branch(n0.df, n0.df_dis, n0.tree_node_dict, self.child_col, self.parent_col, self.branch)

        # Update parent col, select fork and leaf nodes
        df = update_parent_col(df, df_dis, n0.tree_node_dict, self.child_col, self.parent_col, ["fork", "leaf"])
        # csu todo
        # _df1 = df.sort_values(["ID"])
        # _df2 = df.sort_values(["PARENT_ID"])


        # Create features
        # 1. Create length/distance feature
        df_dis = df_dis.drop(["branch"], 1)
        df_dis = df_dis.rename(columns={'descendant': self.child_col})
        # 1.1 Normalize feature
        # len
        max_length = df_dis['len_des_soma'].max()
        df_dis['norm_len'] = df_dis['len'] / max_length
        df_dis['norm_len_des_soma'] = df_dis['len_des_soma'] / max_length
        # direct distance
        max_dis_soma = df_dis['direct_dis_des_soma'].max()
        df_dis['norm_direct_dis_des_soma'] = df_dis['direct_dis_des_soma'] / max_dis_soma
        # 1.2 special features
        df_dis['s/600'] = df_dis['len_des_soma'] / 600
        df_dis['l/18'] = df_dis['len'] / 18
        df_dis['ds/300'] = df_dis['direct_dis_des_soma'] / 300
        # 1.3 create feature for root
        df_dis = df_dis.append(pd.Series([1]+[0]*(len(df_dis.columns)-1), index=df_dis.columns ), ignore_index=True)
        # 1.4 Add feature to df
        col_lst = [self.child_col]+list(self.feature_info.values())
        df_dis = df_dis[df_dis.columns.intersection(col_lst)]
        df = pd.merge(df, df_dis, how="left", on=self.child_col)
        # 2. Create shape feature: ratio_ortho and curvature
        if self.target_level is not None:
            if self.target_level <= 5:
                sys.exit("target_level of the tree is too small (<=5) to create shape_feature! Check Create_Axon_Feature().")
        n1 = Data_Cleaner(self.input_folder, self.nrn_name, "leaf", 5)
        n1 = n1.load_data()
        df = shape_features(df, n1)

        # Add label
        df['label'] = np.where(df[self.type_col] == 2, 1, 0)


        # Rename cols
        _inverse_dict = {y: x for x, y in self.feature_info.items()}
        df = df.rename(columns=_inverse_dict)
        # Subset cols
        cols = self.origin_cols + ["label"] + list(self.feature_info.keys())
        df = df[cols]
        df = df.sort_values(["dp_level", self.child_col], ascending=[False, True])

        # Update data
        self.n0 = n0
        self.n0.df_axon = df
        self.n0.axon_feature = self.feature_info
        return


    def _save_data(self):
        with open(self.fname, "wb") as file:
                pickle.dump([self.n0.df, self.n0.df_dis, self.n0.df_axon, self.n0.df_mix, self.n0.df_level_descend, self.n0.tree_node_dict, self.n0.polarity_dict, self.n0.axon_feature, self.n0.mix_feature], file=file)

        return

    def _change(self):
        with open(self.fname, "wb") as file:
                pickle.dump([self.df, self.df_dis, self.df_axon, self.df_mix, self.df_level_descend, self.tree_node_dict, self.polarity_dict, self.feature_info, self.mix_feature], file=file)

        return


if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for nrn in bar(nrn_lst):
        ax0 = Create_Axon_Feature(input_folder, nrn, remove_method, target_level, overwrite=overwrite)
        ax0 = ax0.load_data()
        # df1 = ax0.df_axon.sort_values(["ID"])
        # df2 = ax0.df_axon.sort_values(["PARENT_ID"])

        # ax0._load_data()
        # ax0._change()

    time.sleep(0.01)


########################################################################################################################
# End of Code
########################################################################################################################
