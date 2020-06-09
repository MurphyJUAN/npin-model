# Purpose:

########################################################################################################################
from .util import *
import tensorflow as tf

########################################################################################################################
# Set up
########################################################################################################################

########################################################################################################################
# Main Code
########################################################################################################################
class DNN:
    def __init__(self, hidden_layer_list, activation_list, trainepochs, batchsize, save_name,
                 threshold=0, weight_init_std_list=0):
        self.a_lst = []
        self.d_lst = []
        self.input_size = 0
        self.output_size = 0
        self.data = {}
        self.save_name = save_name + ".pickle"
        self.h_layer = hidden_layer_list
        self.trainepochs = trainepochs
        self.batchsize = batchsize
        self.th = threshold
        self.w_init = weight_init_std_list
        if type(activation_list) != list:
            if activation_list == 'sigmoid':
                activation_list = ['sigmoid' for i in range(len(self.h_layer))]
            elif activation_list == 'relu':
                activation_list = ['relu' for i in range(len(self.h_layer))]
            elif activation_list == 'selu':
                activation_list = ['selu' for i in range(len(self.h_layer))]
        self.activation = activation_list

    def _init_weight(self, data, label):
        """
            Weight Initialization
            which will depend on what kind of activation function you use
        """
        # parameter initialization
        self.input_size = data.shape[1]
        self.output_size = label.shape[1]

        if type(self.w_init) == list:
            pass
        else:
            all_size_list = [self.input_size] + self.h_layer + [self.output_size]
            weight_init_std_list = []
            for i in range(len(self.activation)):
                if self.activation[i] == 'sigmoid':
                    weight_init_std_list.append(1 / math.sqrt(all_size_list[i]))
                elif self.activation[i] == 'relu':
                    weight_init_std_list.append(2 / math.sqrt(all_size_list[i]))
                elif self.activation[i] == 'selu':
                    weight_init_std_list.append(2 / math.sqrt(all_size_list[i]))
            self.w_init = weight_init_std_list

    def fit(self, data, label):
        """
            the sequence of data and label should be the same

            :param data:
                type: np.array
                shape[0]: number of train node
                shape[1]: features

            :param label:
                type: np.array
                shape[0]: number of train node
                shape[1]: label (0 --> dendrite and 1 --> axon)

            :return:
        """
        # ktc todo protocol of data

        # divide the data into different sets: dendrite and axon
        dendrite_list = []
        axon_list = []
        label = label.reshape(label.shape[0], 1)
        temp = label.tolist()
        for i in range(len(temp)):
            if temp[i][0] == 1:
                temp[i] = [0, 1]
            else:
                temp[i] = [1, 0]
        label = np.array(temp)
        for i in range(len(label)):
            if label[i][0] == 0:
                dendrite_list.append(i)
            elif label[i][0] == 1:
                axon_list.append(i)

        self._init_weight(data, label)

        distribution = [5, 1]
        dendrite_index = data_dist(distribution, len(dendrite_list))
        axon_index = data_dist(distribution, len(axon_list))

        data_set = []
        for i in range(len(dendrite_index)):
            data_set.append([axon_list[j] for j in axon_index[i]] + [dendrite_list[j] for j in dendrite_index[i]])

        # distribute and transform into numpy
        data_input = []
        data_label = []
        for i in range(len(dendrite_index)):
            temp_input, temp_label = list_to_np(data, label, data_set[i])
            data_input.append(temp_input)
            data_label.append(temp_label)

        # save
        data = {}
        data['input'] = data_input
        data['label'] = data_label

        # label of dendrite and axon
        dendrite_list = []
        axon_list = []
        for i in range(len(data_label)):
            temp_dendrite = []
            temp_axon = []
            for j in range(len(data_label[i])):
                if data_label[i][j][0] == 1:
                    temp_dendrite.append(j)
                else:
                    temp_axon.append(j)
            dendrite_list.append(np.array(temp_dendrite))
            axon_list.append(np.array(temp_axon))


        for i in range(sum(distribution)):
            buffer = [j for j in range(sum(distribution))]
            self.data['validation_input'] = data['input'][buffer[0]]
            self.data['validation_label'] = data['label'][buffer[0]]
            buffer.pop(0)
            for j in range(1, distribution[1]):
                self.data['validation_input'] = np.vstack((self.data['validation_input'], data['input'][buffer[0]]))
                self.data['validation_label'] = np.vstack((self.data['validation_label'], data['label'][buffer[0]]))
                buffer.pop(0)
            self.data['train_input'] = data['input'][buffer[0]]
            self.data['train_label'] = data['label'][buffer[0]]
            self.a_lst = axon_list[buffer[0]]
            self.d_lst = dendrite_list[buffer[0]]
            buffer.pop(0)
            for j in range(1, distribution[0]):
                index_correction = len(self.a_lst) + len(self.d_lst)
                self.a_lst = np.hstack((self.a_lst, axon_list[buffer[0]]+index_correction))
                self.d_lst = np.hstack((self.d_lst, dendrite_list[buffer[0]]+index_correction))
                self.data['train_input'] = np.vstack((self.data['train_input'], data['input'][buffer[0]]))
                self.data['train_label'] = np.vstack((self.data['train_label'], data['label'][buffer[0]]))
                buffer.pop(0)

        #self._init_weight(self.data['train_input'], self.data['train_label'])
        self.train_process()

    def train_process(self):
        with tf.device('/cpu:0'):
            # define the components in network
            # csu todo: random initial wieght
            def weight(shape, std):
                return tf.Variable(tf.random_normal(shape, stddev=std), name='W')

            def bias(shape, bias_std=0):
                return tf.Variable(tf.random_normal(shape, stddev=bias_std), name='b')

            def activation_select(input_str):
                if input_str == 'sigmoid':
                    return tf.nn.sigmoid
                elif input_str == 'relu':
                    return tf.nn.relu
                elif input_str == 'selu':
                    return tf.nn.selu

            # construct layers of network
            with tf.name_scope('Input_Layer'):
                x_input = tf.placeholder("float", [None, self.data['train_input'].shape[1]],
                                         name="input")

            c_v = locals()
            with tf.name_scope('Hidden_Layer'):
                with tf.name_scope('Layer_1'):
                    c_v['W1'] = weight([self.data['train_input'].shape[1], self.h_layer[0]], self.w_init[0])    # csu todo: random initial weight
                    c_v['b1'] = bias([1, self.h_layer[0]])
                    activation = activation_select(self.activation[0])
                    output = activation(tf.matmul(x_input, c_v['W1']) + c_v['b1'])
                for i in range(len(self.h_layer) - 1):
                    with tf.name_scope('Layer_' + str(i + 2)):
                        c_v['W' + str(i + 2)] = weight([self.h_layer[i], self.h_layer[i + 1]], self.w_init[i + 1])
                        c_v['b' + str(i + 2)] = bias([1, self.h_layer[i + 1]])
                        activation = activation_select(self.activation[i + 1])
                        output = activation(tf.matmul(output,
                                                      c_v['W' + str(i+2)]) + c_v['b' + str(i+2)])

            with tf.name_scope('Output_Layer'):
                c_v['W' + str(len(self.h_layer)+1)] = weight([self.h_layer[-1], self.data['train_label'].shape[1]],
                                                             std=self.w_init[-1])
                c_v['b' + str(len(self.h_layer)+1)] = bias([1, self.data['train_label'].shape[1]])
                y_predict = tf.matmul(output,
                                      c_v['W' + str(len(self.h_layer)+1)]) + c_v['b' + str(len(self.h_layer)+1)]

            # define loss function
            with tf.name_scope('Loss'):
                y_label = tf.placeholder("float", [None, self.data['train_label'].shape[1]],
                                         name="y_label")
                loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,
                                                                                          labels=y_label),
                                               name='loss_function')

            # optimizer
            with tf.name_scope('Optimizer'):
                lr = tf.Variable(0.01, trainable=False)
                lr_lst = [0.01, 0.006, 0.003]
                optimizer_1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_function)
                optimizer_2 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_function)

            # 確認是否正確預測
            with tf.name_scope('evaluate_model'):
                correct_prediction = tf.equal(tf.argmax(y_label, 1),
                                              tf.argmax(y_predict, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # 參數設定
            totalBatchs = int(self.data['train_input'].shape[0] / self.batchsize)
            if platform == "darwin":
                # OS X
                save_name = 'net/' + 'network_temp'

            elif any([platform == "win32", platform == "win64"]):
                # Windows
                save_name = 'net\\' + 'network_temp' + '.ckpt'

            elif platform == "linux" or platform == "linux2":
                # linux
                save_name = 'net/' + 'network_temp'
            loss_list1 = []
            accuracy_list1 = []
            loss_list2 = []
            accuracy_list2 = []
            t_start = time.time()
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # Training Loop
            max_acc2 = 0
            stage = 0
            sess.run(tf.assign(lr, lr_lst[stage]))
            optimizer = optimizer_1
            epoch_wait = 0
            epoch = -1
            t_set = time.time()
            while (t_set - time.time()) < 3600:
                epoch += 1
                for i in range(totalBatchs):
                    batch_axon = self.a_lst[np.random.choice(len(self.a_lst), int(self.batchsize / 2))]
                    batch_dendrite = self.d_lst[np.random.choice(len(self.d_lst), int(self.batchsize / 2))]
                    batch_mask = np.hstack((batch_axon, batch_dendrite))
                    random.shuffle(batch_mask)
                    x_batch = self.data['train_input'][batch_mask]
                    y_batch = self.data['train_label'][batch_mask]
                    sess.run(optimizer, feed_dict={x_input: x_batch, y_label: y_batch})

                loss1, acc1 = sess.run([loss_function, accuracy],
                                       feed_dict={x_input: self.data['train_input'],
                                                  y_label: self.data['train_label']})
                loss2, acc2 = sess.run([loss_function, accuracy],
                                       feed_dict={x_input: self.data['validation_input'],
                                                  y_label: self.data['validation_label']})

                loss_list1.append(loss1)
                accuracy_list1.append(acc1)
                loss_list2.append(loss2)
                accuracy_list2.append(acc2)

                if epoch == self.trainepochs:
                    print(time.time() - t_set)
                if epoch != 0:
                    if acc2 > self.th:
                        if acc2 > max_acc2:
                            # save model
                            epoch_wait = epoch
                            max_acc2 = acc2
                            saver.save(sess, save_name)

                if epoch % 50 == 0:
                    print("Train Epoch: %02d  Loss_T= %.9f  Loss_V= %.9f  Accuracy_T= %.9f  Accuracy_V= %.9f"
                          % ((epoch + 1), loss1, loss2, acc1, acc2))

                if epoch - epoch_wait > self.trainepochs:
                    print("Train Epoch: %02d  Loss_T= %.9f  Loss_V= %.9f  Accuracy_T= %.9f  Accuracy_V= %.9f"
                          % ((epoch + 1), loss1, loss2, acc1, acc2))
                    stage += 1
                    if stage == len(lr_lst):
                        print(max_acc2)
                        break
                    epoch_wait = epoch
                    saver.restore(sess=sess, save_path=save_name)
                    sess.run(tf.assign(lr, lr_lst[stage]))

            # fine tune
            epoch_wait = epoch
            saver.restore(sess=sess, save_path=save_name)
            stage = 0
            sess.run(tf.assign(lr, lr_lst[stage]))
            optimizer = optimizer_2
            t_set = time.time()
            while (t_set - time.time()) < 3600:
                epoch += 1
                for i in range(totalBatchs):
                    batch_axon = self.a_lst[np.random.choice(len(self.a_lst), int(self.batchsize / 2))]
                    batch_dendrite = self.d_lst[np.random.choice(len(self.d_lst), int(self.batchsize / 2))]
                    batch_mask = np.hstack((batch_axon, batch_dendrite))
                    random.shuffle(batch_mask)
                    x_batch = self.data['train_input'][batch_mask]
                    y_batch = self.data['train_label'][batch_mask]

                    sess.run(optimizer, feed_dict={x_input: x_batch, y_label: y_batch})

                loss1, acc1 = sess.run([loss_function, accuracy],
                                       feed_dict={x_input: self.data['train_input'],
                                                  y_label: self.data['train_label']})

                loss2, acc2 = sess.run([loss_function, accuracy],
                                       feed_dict={x_input: self.data['validation_input'],
                                                  y_label: self.data['validation_label']})

                loss_list1.append(loss1)
                accuracy_list1.append(acc1)
                loss_list2.append(loss2)
                accuracy_list2.append(acc2)
                if epoch != 0:
                    if acc2 > self.th:
                        if acc2 > max_acc2:
                            # save model
                            epoch_wait = epoch
                            max_acc2 = acc2
                            saver.save(sess, save_name)

                if epoch % 50 == 0:
                    print("Train Epoch: %02d  Loss_T= %.9f  Loss_V= %.9f  Accuracy_T= %.9f  Accuracy_V= %.9f"
                          % ((epoch + 1), loss1, loss2, acc1, acc2))

                if epoch - epoch_wait > self.trainepochs:
                    print("Train Epoch: %02d  Loss_T= %.9f  Loss_V= %.9f  Accuracy_T= %.9f  Accuracy_V= %.9f"
                          % ((epoch + 1), loss1, loss2, acc1, acc2))
                    break

            t_end = time.time()
            print(
                "Train Finished takes : %.2f minutes %.2f seconds" % ((t_end - t_start) // 60, (t_end - t_start) % 60))

            # restore the best model (decided by the accuracy of validation data)
            saver.restore(sess=sess, save_path=save_name)

            # run model
            accuracy_validation = sess.run(accuracy,
                                           feed_dict={x_input: self.data['validation_input'],
                                                      y_label: self.data['validation_label']})
            print("Validation Accuracy: %2.2f%%" % (accuracy_validation*100))

            if platform == "darwin":
                # OS X
                save_name = 'net/' + 'epoch_loss'
            elif any([platform == "win32", platform == "win64"]):
                # Windows
                save_name = 'net\\' + 'epoch_loss'
            elif platform == "linux" or platform == "linux2":
                # linux
                save_name = 'net/' + 'epoch_loss'

            # save training precess
            NN_info = {}
            NN_info['loss_list1'] = loss_list1
            NN_info['accuracy_list1'] = accuracy_list1
            NN_info['loss_list2'] = loss_list2
            NN_info['accuracy_list2'] = accuracy_list2
            with open(save_name + time.strftime("%Y-%m-%d_%H%M%S", time.localtime()) + ".pickle", "wb") as file:
                pickle.dump(NN_info, file)

            # save NN
            NN_info = {}
            for i in range(len(self.h_layer) + 1):
                NN_info['W' + str(i + 1)] = sess.run(c_v['W' + str(i + 1)])
                NN_info['b' + str(i + 1)] = sess.run(c_v['b' + str(i + 1)])
            NN_info['NN_list'] = self.activation
            with open(self.save_name, 'wb') as file:
                pickle.dump(NN_info, file)

            sess.close()


    def save_dir(self, new_save_name):
        self.save_name = new_save_name

        return

    def predict_proba(self, data):
        pred = run_NN(self.save_name, data)

        return pred

    def predict(self, data):
        pred = run_NN(self.save_name, data)
        pred = np.where(pred > 0.5, 1, 0)

        return pred

########################################################################################################################
if __name__ == '__main__':
    # parameters
    hidden_layer_list = [10, 10]  # you can choose the numbers of neurons in the NN
    # there are two different ways to set the activation functions we use
    #activation_list = ["selu", "selu"]  # set the activation function of each layer
    activation_list = "selu"  # the activation functions of all the layers will be selu
    trainepochs = 1500  # In the training process, if the precision of model does not improve after 1500 epochs, it will
    # go to the next step.
    batchsize = 600  # the number of input data in one back propagation process
    # ktc todo: should I just assign the folder?
    save_name = "bet1"  # The model will be saved to the path you decide

    N = 10000  # the number of nodes
    D_in = 2**3 - 1  # the number of features in one node
    # the number of the label in one node (should be one)
    # 0 --> dendrite, 1 --> axon
    input_data = np.random.randn(N, D_in)
    input_label = np.random.randint(2, size=(N,))
    print("input data shape: ", input_data.shape)
    print("input label shape: ", input_label.shape)

    # construct the model
    model = DNN(hidden_layer_list, activation_list, trainepochs, batchsize, save_name)
    model.fit(input_data, input_label)

    save_name += ".pickle"
    prediction = run_NN(save_name, input_data)

########################################################################################################################
# End of Code
########################################################################################################################
