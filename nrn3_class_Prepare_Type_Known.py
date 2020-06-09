# Purpose:
# 1. Data point: nrn (represented by first-fork)
# 2. features=[s0, d0, d1,...] by using only reduced level tree & DO NOT drop first fork
# 3. all label = 0 (default)
# 4. Create df with features and label
# 5. Output:
#       prepared_type_leaf5.pkl: prepare all nrns with label = 0.


########################################################################################################################
from util import *
from nrn_params import *
from settings import *
from nrn3_class_Data_Cleaner import Data_Cleaner
from nrn3_class_Prepare_Type import Prepare_Type


########################################################################################################################
# Set up
########################################################################################################################
input_folder = Desktop + '123/'
# nrn_type_lst = ["all", "normal", "small", "multiple", "axon_near"]  # "all", "normal", "small", "multiple", or "axon_near"
remove_method = "leaf"
target_level = 5



########################################################################################################################
# Main Code
########################################################################################################################
class Prepare_Type_Known:

    def __init__(self,
                 input_folder,
                 remove_method=None,
                 target_level=None,
                 origin_cols=['x', 'y', 'z', 'R', 'T', 'ID', 'PARENT_ID'],
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='T'
                 ):

        if all([remove_method is None, target_level is None]):
            self.fname = input_folder+"nrn_cleaned/"+"_".join(["prepared", "type", "known"])+".pkl"
        elif all([type(remove_method) is str, type(target_level) is int]):
            _methodLevel = remove_method + str(target_level)
            self.fname = input_folder+"nrn_cleaned/"+"_".join(["prepared", "type", _methodLevel, "known"])+".pkl"
        else:
            sys.exit("\n remove_method = None or str; target_level = None or int! Check Prepare_Axon.")

        self.input_folder = input_folder
        self.nrn_lst = neuron_dict["all"]
        self.remove_method = remove_method
        self.target_level = target_level
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col

        self.branch = 2


        return


    def load_data(self):
        if(self._is_ready()):
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
            self.type_prepared_df = pickle.load(file)
        return


    def _create_df_from_Data_Cleaner(self):
        print("Prepare Type Known...")
        t0 = Prepare_Type(self.input_folder, self.remove_method, self.target_level)
        t0 = t0.load_data()
        t0 = t0.type_prepared_df
        t0 = t0.drop(["label"], axis=1)

        _, self.L_sort_lst, _ = create_total_L(self.target_level, self.branch)
        df_final = pd.DataFrame({'cols': ['nrn', self.child_col] + self.L_sort_lst + ['label']})

        bar = progressbar.ProgressBar()
        time.sleep(0.01)
        for nrn in bar(self.nrn_lst):
            b1 = bar.value

            ### Read in original df & reduced df from Data_cleaner
            # n0 = Data_Cleaner(self.input_folder, nrn)
            n1 = Data_Cleaner(self.input_folder, nrn, self.remove_method, self.target_level)

            # n0 = n0.load_data()
            n1 = n1.load_data()

            # df0 = n0.df
            # df_dis0 = n0.df_dis

            df1 = n1.df
            # df_dis1 = n1.df_dis

            # tnd0 = neuron_tree_node_dict(df0, self.child_col, self.parent_col)
            # tnd1 = neuron_tree_node_dict(df1, self.child_col, self.parent_col)


            ### Find first fork
            first_fork = find_firstFork_by_LQCol(df1, self.child_col)


            ### Select data points (first-fork only)
            data_points = [first_fork]


            ### Normalize feature


            ### Create feature & label
            # Create df_type1
            df_type1 = df1[[self.child_col, self.type_col]]
            # Expand L to 5 level
            df1 = expand_level_of_L(df1, level=self.target_level)
            # Create total L
            b_lst = detect_branch_of_each_level(df1, L_col='L')
            L_lst, L_sort_lst, n = create_total_L(level=self.target_level, branch=b_lst)

            for idx2, d in enumerate(data_points):
                b2 = idx2
                b_value = b1 * (10**(num_digits(len(self.L_sort_lst)))) + b2

                # Create sub nrn of each data point
                L = df1.loc[df1[self.child_col] == d, 'L'].values[0]
                L_des_lst = find_L_descendant(L, L_lst)
                df_sub = df1.loc[(df1['L_sort'].isin(L_des_lst)) & (df1['Q'] == 0)]
                df_sub = new_L_for_sub_nrn(df_sub, self.child_col, anc_pt=d)

                # Create features=[t0, t1,...] & label
                # df_sub = pd.merge(df_sub, df_type1, on=self.child_col, how='left')
                # s_0 = df_sub.loc[df_sub[self.child_col] == d, 'norm_len_des_soma'].values[0]
                label = 0

                df_id = pd.DataFrame({'cols': ['nrn', self.child_col], b_value: [nrn, d]})
                # df_s = pd.DataFrame({'cols': 's_0', b_value: s_0}, index=[0])
                df_t = pd.DataFrame({'cols': df_sub['L_sort'], b_value: df_sub[self.type_col]})
                df_l = pd.DataFrame({'cols': 'label', b_value: label}, index=[0])
                df_temp = pd.concat([df_id, df_t, df_l])

                df_final = pd.merge(df_final, df_temp, how='left', on='cols')

        time.sleep(0.01)

        ### Create df_final (transpose, colnames, fillna(0), label={'axon':1, 'dendrite':0})
        df_final = df_final.T.reset_index(drop=True)
        df_final.columns = df_final.iloc[0]
        df_final = df_final[1:].reset_index(drop=True)
        for i in self.L_sort_lst:
            df_final = df_final.rename(columns={i: "t_" + i})
        df_final = df_final.fillna(0)
        df_final = pd.merge(t0, df_final, on=["nrn", self.child_col], how="left")

        self.type_prepared_df = df_final

        return


    def _save_data(self):
        with open(self.fname, "wb") as file:
            pickle.dump(self.type_prepared_df, file=file)
        return


if __name__ == '__main__':
    type0 = Prepare_Type_Known(input_folder, remove_method, target_level)

    t0 = type0.load_data()

    df = t0.type_prepared_df

    a = 123


########################################################################################################################
# End of Code
########################################################################################################################
