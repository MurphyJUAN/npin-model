# Purpose:
# 1. Data point: each node in level tree WITHOUT soma
# 2. features=[s0, d0, d1,...] by using only reduced level tree & DO NOT drop first fork
# 3. label={'axon':1, 'dendrite':0}
# 4. Create df with features and label
# 5. Output:
#       prepared_axon_leaf5.pkl: prepare nrns.


########################################################################################################################
from .util import *
from .nrn_params import *
from .settings import *
from .nrn3_class_Create_Axon_Feature import Create_Axon_Feature


########################################################################################################################
# Set up
########################################################################################################################
input_folder = 'data/'
nrn_type = ["new_all_2"]
# nrn_type = ["test"]
# remove_method = "leaf"
# target_level = 5
remove_method = None
target_level = None
augment_number = 10     # int >= 0
overwrite = False

# csu todo
# remove_method = None
# target_level = None

########################################################################################################################
# Main Code
########################################################################################################################
class Prepare_Axon:

    def __init__(self,
                 input_folder,
                 nrn_type,
                 remove_method=None,
                 target_level=None,
                 augment_number=0,
                 child_col='ID',
                 parent_col='PARENT_ID',
                 type_col='type_pre',  # Here use pre_relabel type 'type_pre'; the original type is 'T'.
                 overwrite=False
                 ):
        self.sys_path = os.path.dirname(__file__)
        self.input_folder = input_folder
        self.nrn_type = sorted(nrn_type)
        self.remove_method = remove_method
        self.target_level = target_level
        self.augmant_number = augment_number
        self.child_col = child_col
        self.parent_col = parent_col
        self.type_col = type_col
        self.overwrite = overwrite

        # fname
        self.fname_dict = {}
        for i in ["forEvaluate", "forTrain", "forTrainFk"]:
            _fname0 = "_".join(["prepare", i])

            # _trainTest
            _trainTest = '&'.join(self.nrn_type)
            _fname0 = "_".join([_fname0, _trainTest])

            # _methodLevel
            if all([type(remove_method) is str, type(target_level) is int]):
                _methodLevel = remove_method + str(target_level)
                _fname0 = "_".join([_fname0, _methodLevel])

            # _augment
            if i == 'forTrainFk':
                if augment_number == 0:
                    continue
                elif type(augment_number) is int:
                    _augment = "fk" + str(augment_number)
                    _fname0 = "_".join([_fname0, _augment])
                else:
                    sys.exit("\n augment_nrn_type = None or list; augment_number = int! Check Prepare_Axon.")

            self.fname_dict[i] = self.sys_path + input_folder + "nrn_cleaned/" + _fname0 + ".pkl"


        # nrn_lst
        self.nrn_lst = []
        if isinstance(nrn_type, list):
            for t in nrn_type:
                self.nrn_lst += neuron_dict[t]

        # Create fake data name
        self.fk_nrn_name_lst = []
        if augment_number > 0:
            for nrn_name in self.nrn_lst:
                self.fk_nrn_name_lst += ["fk" + str(i) + "_" + nrn_name for i in range(augment_number)]




        return


    def load_data(self, lst=["forEvaluate", "forTrain", "forTrainFk"]):
        lst = list(set(self.fname_dict.keys()))
        self.fname_dict = {key: self.fname_dict[key] for key in lst}
        if all([self._is_ready(), not self.overwrite]):
            self._load_data()
        else:
            self._create_df_from_Create_Axon_Feature()
            self._load_data()
        return self


    def _is_ready(self):
        for k, fname in self.fname_dict.items():
            if k == "forTrainFk":
                if os.path.exists(fname):
                    continue
                else:
                    _lst = get_fileNames_from_directory("./data/nrn_cleaned/", "pkl", True)
                    if any('_fk' in x for x in _lst):
                        _n = max([int(x.split('_fk')[1]) for x in _lst if k in x])
                        if _n > self.augmant_number:
                            _fk = "".join(["fk", str(_n)])+".pkl"
                            fname = "_".join([fname.split('_fk')[0], _fk])
                            self.fname_dict[k] = fname
            if os.path.exists(fname):
                x = True
            else:
                x = False
                break
        return x


    def _load_data(self):
        self.prepare_dict = {}
        for k, v in self.fname_dict.items():
            with open(v, "rb") as file:
                self.prepare_dict[k] = pickle.load(file)
        return


    def _create_df_from_Create_Axon_Feature(self):
        for k in self.fname_dict.keys():
            if k != "forTrainFk":
                # original level tree(for post_relabel) & axon(for training)
                if all([os.path.exists(self.fname_dict[k]), not self.overwrite]):
                    continue

                print("prepare dfs 'for evaluate' & 'for train' ...")
                start_time = time.time()
                df0 = None
                df1 = None
                for nrn in self.nrn_lst:
                    n1 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level)
                    n1 = n1.load_data()
                    _df0 = update_parent_col(n1.df, n1.df_dis, n1.tree_node_dict, "ID", "PARENT_ID")
                    # _df0 = update_parent_col(n1.df_axon, n1.df_dis, n1.tree_node_dict, "ID", "PARENT_ID")
                    _df1 = n1.df_axon
                    if df0 is None:
                        df0 = _df0
                    else:
                        df0 = df0.append(_df0)

                    if df1 is None:
                        df1 = _df1
                    else:
                        df1 = df1.append(_df1)

                    del n1, _df0, _df1
                    gc.collect()

                df0 = df0.sort_values(['nrn', 'ID']).reset_index(drop=True)
                df0 = df0.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'type_pre']]
                with open(self.fname_dict["forEvaluate"], "wb") as file:
                    pickle.dump(df0, file=file)

                df1 = df1.sort_values(['nrn', 'ID']).reset_index(drop=True)
                with open(self.fname_dict["forTrain"], "wb") as file:
                    pickle.dump(df1, file=file)

                print("Elapsed Time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))



            else:
                # fk(for training)
                if all([os.path.exists(self.fname_dict[k]), not self.overwrite]):
                    continue

                print("prepare df 'for train fk'...")
                start_time = time.time()
                df = None
                for nrn in self.fk_nrn_name_lst:
                    n1 = Create_Axon_Feature(self.input_folder, nrn, self.remove_method, self.target_level)
                    n1 = n1.load_data()
                    _df = n1.df_axon
                    if df is None:
                        df = _df
                    else:
                        df = df.append(_df)
                    del n1, _df
                    gc.collect()
                df = df.sort_values(['nrn', 'ID']).reset_index(drop=True)
                with open(self.fname_dict[k], "wb") as file:
                    pickle.dump(df, file=file)
                print("Elapsed Time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

        return



if __name__ == '__main__':
    ax0 = Prepare_Axon(input_folder, nrn_type, remove_method, target_level, augment_number, overwrite=overwrite)

    # ax0 = ax0.load_data()

    ax0 = ax0.load_data(["forTrainFk"])

    # ax0 = ax0.load_data(["forTrain", "forTrainFk"])

    # ax0 = ax0.load_data(["forEvaluate"])

    # a = 123

    # ax0._load_data()
    # ax0._create_db()
    # ax0._save_data()

    # df = df[["nrn", "ID"]]

    # _method_level = remove_method+str(target_level)
    # a0.axon_prepared_df.to_csv(input_folder + "_".join(["axon", _method_level])+'.csv', index=False)
    # df.to_csv(input_folder + _method_level + '.csv', index=False)


########################################################################################################################
# End of Code
########################################################################################################################
