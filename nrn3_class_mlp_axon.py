# Purpose:
# 1. Find axon
# 2. Test: 0.3 normal, Train: 0.7 normal
# 3. Use deep learning to classify: mlp


########################################################################################################################
from util import *
from nrn_params import *
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dropout
################################################################################
def show_train_history(train_history, train, validation, show_plt=False):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if show_plt:
        plt.show()

# -------------------------------------------------------------------------------
def plot_images_labels_prediction(images, labels, prediction, idx, num=10, show_plt=False):
        fig = plt.gcf()
        fig.set_size_inches(12, 14)
        if num > 25:
            num = 25
        for i in range(0, num):
            ax = plt.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = "label=" + str(labels[idx])
            if len(prediction) > 0:
                title += ", predict=" + str(prediction[idx])

            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        if show_plt:
            plt.show()


################################################################################


########################################################################################################################
# Set up
########################################################################################################################
origin_cols = ['x',  'y',  'z',  'R',  'T',  'ID',  'PARENT_ID']
child_col = 'ID'
parent_col = 'PARENT_ID'
type_col = 'T'
decimal = 0
removes = 20
dis_depend_on_lst = ['leaf']     # ['all', 'leaf']
kde = True
log_y = False
input_folder_csv = '/Users/csu/Desktop/Neuron/nrn_result/'
output_folder_plot = '/Users/csu/Desktop/Neuron/nrn_plot/'


nrn_type = "pcb"
feature_type = "af8"

# Choose model


########################################################################################################################
# Main Code
########################################################################################################################
### Train & Test
print("Train & Test: \n")
random.seed(123)

# Load data
# df_final = df_dis0 = pd.read_csv(input_folder_csv + 'axon_feature_label.csv')
# _, L_sort_lst, _ = create_total_L(level=5, branch=2)
# feature_lst = ['s0'] + L_sort_lst

with open(Desktop + "123/nrn_cleaned/prepared_axon.pkl", 'rb') as f:
    df_final = pickle.load(f)

df_final = df_final.loc[df_final["nrn"].isin(neuron_dict[nrn_type])].reset_index(drop=True)

feature_dict = create_axon_feature_dict(5, 2)
feature_lst = feature_dict[feature_type]
input_num = len(feature_lst)


msk = np.random.rand(len(df_final)) < 0.7
train_df = df_final[msk]
test_df = df_final[~msk]

x_train_df = train_df[feature_lst]
x_test_df = test_df[feature_lst]
y_train_label = train_df['label'].values
y_test_label = test_df['label'].values


# Reshape (the feature already normalized)
x_Train_normalize = x_train_df.values
x_Test_normalize = x_test_df.values


# One-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

################################################################################

# Create linear cumulative model
model = Sequential()

# Create input layer and hidden layer
model.add(Dense(units=5,
                input_dim=input_num,
                kernel_initializer='normal',
                activation='relu'))

# # Add drop out to avoid overfitting
# model.add(Dropout(0.5))

# Create hidden layer2
model.add(Dense(units=2,
                kernel_initializer='normal',
                activation='relu'))

# # Create hidden layer3
# model.add(Dense(units=2,
#                 kernel_initializer='normal',
#                 activation='relu'))
#
# # Add drop out to avoid overfitting
# model.add(Dropout(0.5))

# Create output layer
model.add(Dense(units=2,
                kernel_initializer='normal',
                activation='softmax'))

# Summarize model
print(model.summary())


# Loss function, optimizer, and measurement settings
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot, validation_split=0.2,
                          epochs=300, batch_size=50, verbose=2)

# Show train history
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

################################################################################

# Evaluate model accuracy among Test set
time.sleep(0.01)
scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print('accuracy=', scores[1])

# Predict Test set
prediction = model.predict_classes(x_Test_normalize)
print('predictions:', prediction[:10])

# Show prediction result
# plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340, num=10, show_plt=False)

################################################################################

# Confusion matrix
c_matrix = pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
print('Confusion matrix: \n', c_matrix)

'''
# List out wrong prediction
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
print('df[:2]:\n', df[:2])
print('label=5 & pred=3:\n', df[(df.label == 5) & (df.predict == 3)])
# print('label=5 & pred!=5:\n', df[(df.label == 5) & (df.predict != 5)])

# Look up the picture of wrong prediction
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340, num=1, show_plt=True)
'''

########################################################################################################################
