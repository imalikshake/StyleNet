import tensorflow as tf
from data_util import BatchGenerator
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff()

class GenreLSTM(object):
    def __init__(self, dirs, mini=False, bi=False, one_hot=True, input_size=176, output_size=88, num_layers=3, batch_count=8):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.batch_count = int(batch_count)
        self.dirs = dirs
        self.bi = bi
        self.mini = mini
        self.one_hot = one_hot


    def prepare_bidiretional(self, glorot=True):
        print("[*] Preparing bidirectional dynamic RNN...")
        self.input_cell = tf.contrib.rnn.LSTMCell(self.input_size, forget_bias=1.0)
        self.input_cell = tf.contrib.rnn.DropoutWrapper(self.input_cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
        self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(self.input_cell, self.inputs, sequence_length=self.seq_len, dtype=tf.float32)


        with tf.variable_scope("encode") as scope:

            self.j_cell_fw = tf.contrib.rnn.LSTMBlockCell(self.input_size,forget_bias=1.0)
            self.j_cell_fw = tf.contrib.rnn.DropoutWrapper(self.j_cell_fw, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

            self.j_cell_bw = tf.contrib.rnn.LSTMBlockCell(self.input_size,forget_bias=1.0)
            self.j_cell_bw = tf.contrib.rnn.DropoutWrapper(self.j_cell_bw, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

            if self.num_layers > 1:
                self.j_cell_fw = tf.contrib.rnn.MultiRNNCell([self.j_cell_fw]*self.num_layers)
                self.j_cell_bw = tf.contrib.rnn.MultiRNNCell([self.j_cell_bw]*self.num_layers)



            # self.j_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            (self.j_fw, self.j_bw) , _ = tf.nn.bidirectional_dynamic_rnn(
                                                        self.j_cell_fw,
                                                        self.j_cell_bw,
                                                        self.enc_outputs,
                                                        sequence_length=self.seq_len,
                                                        dtype=tf.float32)


            self.jazz_outputs  = tf.concat([self.j_fw, self.j_bw],2)
            # self.jazz_outputs = tf.add(self.j_outputs[0], self.j_outputs[1])

            scope.reuse_variables()


            self.c_cell_fw = tf.contrib.rnn.LSTMBlockCell(self.input_size,forget_bias=1.0)
            self.c_cell_fw = tf.contrib.rnn.DropoutWrapper(self.c_cell_fw, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

            self.c_cell_bw = tf.contrib.rnn.LSTMBlockCell(self.input_size,forget_bias=1.0)
            self.c_cell_bw = tf.contrib.rnn.DropoutWrapper(self.c_cell_bw, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

            if self.num_layers > 1:
                self.c_cell_fw  = tf.contrib.rnn.MultiRNNCell([self.c_cell_fw ]*self.num_layers)
                self.c_cell_bw  = tf.contrib.rnn.MultiRNNCell([self.c_cell_bw ]*self.num_layers)

            (self.c_fw, self.c_bw), _ =  tf.nn.bidirectional_dynamic_rnn(
            # self.c_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                                                        self.c_cell_fw,
                                                        self.c_cell_bw,
                                                        self.enc_outputs,
                                                        sequence_length=self.seq_len,
                                                        dtype=tf.float32)


            self.classical_outputs  = tf.concat([self.c_fw, self.c_bw],2)

            # self.classical_outputs = tf.add(self.c_outputs[0], self.c_outputs[1])


        self.jazz_B = tf.Variable(tf.random_normal([self.output_size], stddev=0.1))
        self.classical_B = tf.Variable(tf.random_normal([self.output_size], stddev=0.1))

        if glorot:
            self.jazz_W = tf.get_variable("jazz_W", shape=[self.input_size*2, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            self.classical_W = tf.get_variable("classical_W", shape=[self.input_size*2, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.jazz_W = tf.Variable(tf.random_normal([self.input_size*2,self.output_size], stddev=0.1))
            self.classical_W = tf.Variable(tf.random_normal([self.input_size*2,self.output_size], stddev=0.1))

        self.jazz_linear_out = tf.reshape(self.jazz_outputs, [tf.shape(self.true_jazz_outputs)[0]*self.seq_len[-1], 2*self.input_size])
        self.jazz_linear_out = tf.matmul(self.jazz_linear_out, self.jazz_W) + self.jazz_B
        self.jazz_linear_out = tf.reshape(self.jazz_linear_out,[tf.shape(self.true_jazz_outputs)[0],tf.shape(self.true_jazz_outputs)[1], tf.shape(self.true_jazz_outputs)[2]])

        self.classical_linear_out = tf.reshape(self.classical_outputs, [tf.shape(self.true_classical_outputs)[0]*self.seq_len[-1], 2*self.input_size])
        self.classical_linear_out = tf.matmul(self.classical_linear_out, self.classical_W) + self.classical_B
        self.classical_linear_out = tf.reshape(self.classical_linear_out,[tf.shape(self.true_classical_outputs)[0],tf.shape(self.true_classical_outputs)[1], tf.shape(self.true_classical_outputs)[2]])

    def prepare_unidiretional(self, glorot=True):
        print("[*] Preparing unidirectional dynamic RNN...")
        self.input_cell = tf.contrib.rnn.LSTMCell(self.input_size, forget_bias=1.0)
        self.input_cell = tf.contrib.rnn.DropoutWrapper(self.input_cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
        self.enc_outputs, self.enc_states = tf.nn.dynamic_rnn(self.input_cell, self.inputs, sequence_length=self.seq_len, dtype=tf.float32)

        with tf.variable_scope("encode") as scope:

            self.jazz_cell = tf.contrib.rnn.LSTMCell(self.input_size, forget_bias=1.0)
            self.jazz_cell = tf.contrib.rnn.DropoutWrapper(self.jazz_cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

            self.jazz_outputs, self.jazz_states = tf.nn.dynamic_rnn(self.jazz_cell, self.enc_outputs, sequence_length=self.seq_len, dtype=tf.float32)

            scope.reuse_variables()

            self.classical_cell = tf.contrib.rnn.LSTMCell(self.input_size, forget_bias=1.0)
            self.classical_cell = tf.contrib.rnn.DropoutWrapper(self.classical_cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
            self.classical_outputs, self.classical_states = tf.nn.dynamic_rnn(self.classical_cell, self.enc_outputs, sequence_length=self.seq_len, dtype=tf.float32)

        # self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)
        # self.stacked_cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)

        self.jazz_B = tf.Variable(tf.random_normal([self.output_size], stddev=0.1))
        self.classical_B = tf.Variable(tf.random_normal([self.output_size], stddev=0.1))

        if glorot:
            self.jazz_W = tf.get_variable("jazz_W", shape=[self.input_size, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            self.classical_W = tf.get_variable("classical_W", shape=[self.input_size, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.jazz_W = tf.Variable(tf.random_normal([self.input_size,self.output_size], stddev=0.1))
            self.classical_W = tf.Variable(tf.random_normal([self.input_size,self.output_size], stddev=0.1))

        self.jazz_linear_out = tf.reshape(self.jazz_outputs, [tf.shape(self.true_jazz_outputs)[0]*self.seq_len[-1], self.input_size])
        self.jazz_linear_out = tf.matmul(self.jazz_linear_out, self.jazz_W) + self.jazz_B
        self.jazz_linear_out = tf.reshape(self.jazz_linear_out,[tf.shape(self.true_jazz_outputs)[0],tf.shape(self.true_jazz_outputs)[1], tf.shape(self.true_jazz_outputs)[2]])

        self.classical_linear_out = tf.reshape(self.classical_outputs, [tf.shape(self.true_classical_outputs)[0]*self.seq_len[-1], self.input_size])
        self.classical_linear_out = tf.matmul(self.classical_linear_out, self.classical_W) + self.classical_B
        self.classical_linear_out = tf.reshape(self.classical_linear_out,[tf.shape(self.true_classical_outputs)[0],tf.shape(self.true_classical_outputs)[1], tf.shape(self.true_classical_outputs)[2]])

    def prepare_model(self, bi=False):

        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size])

        self.true_jazz_outputs = tf.placeholder(tf.float32, [None, None, self.output_size])
        self.true_classical_outputs = tf.placeholder(tf.float32, [None, None, self.output_size])

        self.seq_len = tf.placeholder(tf.int32, [None])

        self.input_keep_prob = tf.placeholder(tf.float32, None)
        self.output_keep_prob = tf.placeholder(tf.float32, None)

        if self.bi:
            self.prepare_bidiretional()
        else:
            self.prepare_unidiretional()

        self.jazz_negation = tf.subtract(self.true_jazz_outputs, self.jazz_linear_out)
        self.classical_negation = tf.subtract(self.true_classical_outputs, self.classical_linear_out)

        self.jazz_loss =  tf.reduce_mean(tf.square(tf.subtract(self.jazz_linear_out, self.true_jazz_outputs)))
        self.classical_loss =  tf.reduce_mean(tf.square(tf.subtract(self.classical_linear_out, self.true_classical_outputs)))

        tf.summary.scalar("Jazz error", self.jazz_loss)
        tf.summary.scalar("Classical error", self.classical_loss)
        tf.summary.scalar("Average error", self.jazz_loss+self.classical_loss/2)

        tf.summary.histogram("Jazz negation", self.jazz_negation)
        tf.summary.histogram("Classical negation", self.classical_negation)

    def clip_optimizer(self, learning_rate, loss):
        opt = tf.train.AdamOptimizer(learning_rate)
        gradients = opt.compute_gradients(loss)

        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 10), var)

        return opt.apply_gradients(gradients)

    def train(self, data, model=None, starting_epoch=0, clip_grad=True, epochs=1001, input_keep_prob=0.5, output_keep_prob=0.5, learning_rate=0.001 , eval_epoch=20,val_epoch=10, save_epoch=1):

        self.data = data

        if clip_grad:
            jazz_optimizer = self.clip_optimizer(learning_rate,self.jazz_loss)
            classical_optimizer = self.clip_optimizer(learning_rate,self.classical_loss)
        else:
            jazz_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.jazz_loss)
            classical_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.classical_loss)



        self.sess = tf.Session()

        self.c_in_list, self.c_out_list,self.c_input_lens, self.c_files = self.eval_set('classical')
        self.j_in_list, self.j_out_list,self.j_input_lens, self.j_files = self.eval_set('jazz')

        if model:
            self.load(model)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.summary_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'train'), graph=self.sess.graph_def)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'test'), graph=self.sess.graph_def)

        classical_batcher = BatchGenerator(self.data["classical"]["X"], self.data["classical"]["Y"], self.batch_count, self.input_size, self.output_size, mini=self.mini)
        jazz_batcher = BatchGenerator(self.data["jazz"]["X"], self.data["jazz"]["Y"], self.batch_count, self.input_size, self.output_size, mini=self.mini)

        self.v_classical_batcher = self.validate("classical")
        self.v_classical_batcher = self.v_classical_batcher.batch()

        self.v_jazz_batcher = self.validate("jazz")
        self.v_jazz_batcher = self.v_jazz_batcher.batch()


        classical_generator = classical_batcher.batch()
        jazz_generator = jazz_batcher.batch()

        print("[*] Initiating training...")

        for epoch in xrange(starting_epoch, epochs):

            classical_epoch_avg = 0
            jazz_epoch_avg = 0

            print("[*] Epoch %d" % epoch)
            for batch in range(classical_batcher.batch_count):
                batch_X, batch_Y, batch_len = classical_generator.next()
                batch_len = [batch_len] * len(batch_X)
                epoch_error, classical_summary, _  =  self.sess.run([self.classical_loss,
                                                         self.summary_op,
                                                         classical_optimizer,
                                                         ], feed_dict={ self.inputs: batch_X,
                                                                                  self.true_classical_outputs: batch_Y,
                                                                                  self.true_jazz_outputs: batch_Y,
                                                                                  self.seq_len: batch_len,
                                                                                  self.input_keep_prob: input_keep_prob,
                                                                                  self.output_keep_prob: output_keep_prob})
                classical_epoch_avg += epoch_error
                print("\tBatch %d/%d, Training MSE for Classical batch: %.9f" % (batch+1, classical_batcher.batch_count, epoch_error))
                self.train_writer.add_summary(classical_summary, epoch*classical_batcher.batch_count + epoch)

            for batch in range(jazz_batcher.batch_count):
                batch_X, batch_Y, batch_len = jazz_generator.next()
                batch_len = [batch_len] * len(batch_X)
                epoch_error, jazz_summary, _  =  self.sess.run([self.jazz_loss,
                                                         self.summary_op,
                                                         jazz_optimizer,
                                                         ], feed_dict={ self.inputs: batch_X,
                                                                                  self.true_jazz_outputs: batch_Y,
                                                                                  self.true_classical_outputs: batch_Y,
                                                                                  self.seq_len: batch_len,
                                                                                  self.input_keep_prob: input_keep_prob,
                                                                                  self.output_keep_prob: output_keep_prob})
                jazz_epoch_avg += epoch_error
                print("\tBatch %d/%d, Training MSE for Jazz batch: %.9f" % (batch+1, jazz_batcher.batch_count, epoch_error))

                self.train_writer.add_summary(jazz_summary, epoch*jazz_batcher.batch_count + epoch)
                # self.validation(epoch)

            print("[*] Average Training MSE for Classical epoch %d: %.9f" % (epoch, classical_epoch_avg/classical_batcher.batch_count))
            print("[*] Average Training MSE for Jazz epoch %d: %.9f" % (epoch, jazz_epoch_avg/jazz_batcher.batch_count))

            if epoch % val_epoch == 0 :
                print("[*] Validating model...")
                self.validation(epoch)

            if epoch % save_epoch == 0 :
                self.save(epoch)

            if epoch % eval_epoch == 0 :
                print("[*] Evaluating model...")
                self.evaluate(epoch)

        print("[*] Training complete.")

    def load(self, model_name, path=None) :
        print(" [*] Loading checkpoint...")
        self.saver = tf.train.Saver(max_to_keep=0)
        if not path:
            self.saver.restore(self.sess, os.path.join(self.dirs['model_path'], model_name))
        else:
            self.sess = tf.Session()
            self.saver.restore(self.sess, path)

    def save(self, epoch):
        print("[*] Saving checkpoint...")
        model_name =  "model-e%d.ckpt" % (epoch)
        self.saver = tf.train.Saver(max_to_keep=0)
        save_path = self.saver.save(self.sess, os.path.join(self.dirs['model_path'], model_name))
        print("[*] Model saved in file: %s" % save_path)

    def predict(self, input_path, output_path):
        in_list = []
        out_list = []
        filenames = []
        input_lens = []

        loaded = np.load(input_path)
        true_vel = np.load(output_path)/127

        in_list.append(loaded)
        out_list.append(true_vel)

        input_len = [len(loaded)]

        c_error, c_out, j_out, e_out = self.sess.run([self.classical_loss, self.classical_linear_out, self.jazz_linear_out, self.enc_outputs], feed_dict={self.inputs:in_list,
                                                                                                                                                            self.seq_len:input_len,
                                                                                                                                                            self.input_keep_prob:1.0,
                                                                                                                                                            self.output_keep_prob:1.0,
                                                                                                                                                            self.true_classical_outputs:out_list,
                                                                                                                                                            self.true_jazz_outputs:out_list})

        return c_error, c_out, j_out, e_out, out_list

    def validate(self, type):
        '''Handles validation set data'''
        input_eval_path = os.path.join(self.dirs['eval_path'], "inputs")
        vel_eval_path = os.path.join(self.dirs['eval_path'], "velocities")

        c_input_eval_path = os.path.join(input_eval_path, "classical")
        c_vel_eval_path = os.path.join(vel_eval_path, "classical")

        j_input_eval_path = os.path.join(input_eval_path, "jazz")
        j_vel_eval_path = os.path.join(vel_eval_path, "jazz")

        if type == "classical":
            input_folder = os.listdir(c_input_eval_path)
            file_count = len(input_folder)
            vel_eval_path = c_vel_eval_path
            input_eval_path = c_input_eval_path
        else:
            input_folder = os.listdir(j_input_eval_path)
            file_count = len(input_folder)
            vel_eval_path = j_vel_eval_path
            input_eval_path = j_input_eval_path
            #CLASSICS

        in_list = []
        out_list = []
        filenames = []
        for i, filename in enumerate(input_folder):
            if filename.split('.')[-1] == 'npy':

                vel_path = os.path.join(vel_eval_path, filename)
                input_path = os.path.join(input_eval_path, filename)

                true_vel = np.load(vel_path)/127
                loaded = np.load(input_path)

                if not self.one_hot:
                    loaded = loaded/2

                in_list.append(loaded)
                out_list.append(true_vel)
                filenames.append(filename)
        valid_generator = BatchGenerator(in_list, out_list, self.batch_count, self.input_size, self.output_size, mini=False)
        return valid_generator

    def validation(self, epoch, pred_save=False):
        '''Computes and logs loss of validation set'''
        in_list, out_list, input_len = self.v_classical_batcher.next()
        input_len = [input_len] * len(in_list)
        c_error, c_out, j_out, e_out, c_summary = self.sess.run([self.classical_loss,
                                          self.classical_linear_out,
                                          self.jazz_linear_out,
                                          self.enc_outputs,
                                          self.summary_op],

                                        feed_dict={self.inputs:in_list,
                                                   self.seq_len:input_len,
                                                   self.input_keep_prob:1.0,
                                                   self.output_keep_prob:1.0,
                                                   self.true_classical_outputs:out_list,
                                                   self.true_jazz_outputs:out_list})


        # for i, x in enumerate(c_out):
            # self.plot_evaluation(epoch, c_files[i], c_out[i], j_out[i], e_out[i], out_list[i])

        in_list, out_list, input_len = self.v_jazz_batcher.next()
        input_len = [input_len] * len(in_list)

        j_error, j_out, c_out, e_out, j_summary = self.sess.run([self.jazz_loss,
                                          self.jazz_linear_out,
                                          self.classical_linear_out,
                                          self.enc_outputs,
                                          self.summary_op],

                                        feed_dict={self.inputs:in_list,
                                                   self.seq_len:input_len,
                                                   self.input_keep_prob:1.0,
                                                   self.output_keep_prob:1.0,
                                                   self.true_jazz_outputs:out_list,
                                                   self.true_classical_outputs:out_list})


        # for i, x in enumerate(c_out):
            # self.plot_evaluation(epoch, j_files[i], c_out[i], j_out[i], e_out[i], out_list[i])

        # print("[*] Validating Model...")

        print("[*] Average Test MSE for Classical epoch %d: %.9f" % (epoch, c_error))
        print("[*] Average Test MSE for Jazz epoch %d: %.9f" % (epoch, j_error))


        self.test_writer.add_summary(j_summary, epoch)
        self.test_writer.add_summary(c_summary, epoch)

    def eval_set(self, type):
        '''Loads validation set.'''
        input_eval_path = os.path.join(self.dirs['eval_path'], "inputs")
        vel_eval_path = os.path.join(self.dirs['eval_path'], "velocities")

        c_input_eval_path = os.path.join(input_eval_path, "classical")
        c_vel_eval_path = os.path.join(vel_eval_path, "classical")

        j_input_eval_path = os.path.join(input_eval_path, "jazz")
        j_vel_eval_path = os.path.join(vel_eval_path, "jazz")

        if type == "classical":
            input_folder = os.listdir(c_input_eval_path)
            file_count = len(input_folder)
            vel_eval_path = c_vel_eval_path
            input_eval_path = c_input_eval_path
        else:
            input_folder = os.listdir(j_input_eval_path)
            file_count = len(input_folder)
            vel_eval_path = j_vel_eval_path
            input_eval_path = j_input_eval_path
            #CLASSICS

        in_list = []
        out_list = []
        filenames = []
        input_lens = []

        for i, filename in enumerate(input_folder):
            if filename.split('.')[-1] == 'npy':

                vel_path = os.path.join(vel_eval_path, filename)
                input_path = os.path.join(input_eval_path, filename)

                true_vel = np.load(vel_path)/120
                loaded = np.load(input_path)

                if not self.one_hot:
                    loaded = loaded/2

                in_list.append([loaded])
                out_list.append([true_vel])
                filenames.append(filename)
                input_len = [len(loaded)]
                input_lens.append(input_len)

        return in_list, out_list, input_lens, filenames

    def evaluate(self, epoch, pred_save=False):
        '''Performs prediciton and plots results on validation set.'''
        for i, filename in enumerate(self.c_files):
            c_error, c_out, j_out, e_out, summary = self.sess.run([self.classical_loss,
                                              self.classical_linear_out,
                                              self.jazz_linear_out,
                                              self.enc_outputs,
                                              self.summary_op],

                                            feed_dict={self.inputs:self.c_in_list[i],
                                                       self.seq_len:self.c_input_lens[i],
                                                       self.input_keep_prob:1.0,
                                                       self.output_keep_prob:1.0,
                                                       self.true_classical_outputs:self.c_out_list[i],
                                                       self.true_jazz_outputs:self.c_out_list[i]})


            self.plot_evaluation(epoch, filename, c_out, j_out, e_out, self.c_out_list[i])
                # if pred_save:
                #     predicted = os.path.join(self.dirs['pred_path'], filename.split('.')[0] + "-e%d" % (epoch)+".npy")
                #     np.save(predicted, linear[-1])

        for i, filename in enumerate(self.j_files):
            j_error, j_out, c_out, e_out, summary = self.sess.run([self.jazz_loss,
                                              self.jazz_linear_out,
                                              self.classical_linear_out,
                                              self.enc_outputs,
                                              self.summary_op],

                                            feed_dict={self.inputs:self.j_in_list[i],
                                                       self.seq_len:self.j_input_lens[i],
                                                       self.input_keep_prob:1.0,
                                                       self.output_keep_prob:1.0,
                                                       self.true_classical_outputs:self.j_out_list[i],
                                                       self.true_jazz_outputs:self.j_out_list[i]})

            self.plot_evaluation(epoch, filename, c_out, j_out, e_out, self.j_out_list[i])
                # if pred_save:
                #     predicted = os.path.join(self.dirs['pred_path'], filename.split('.')[0] + "-e%d" % (epoch)+".npy")
                #     np.save(predicted, linear[-1])


    def plot_evaluation(self, epoch, filename, c_out, j_out, e_out, out_list, path=None):
        '''Plotting/Saving training session graphs
        epoch -- epoch number
        c_out -- classical output
        j_out -- jazz output
        e_out -- interpretation layer output
        out_list -- actual output
        output_size -- output width
        path -- Save path'''

        fig = plt.figure(figsize=(14,11), dpi=120)
        fig.suptitle(filename, fontsize=10, fontweight='bold')

        graph_items = [out_list[-1]*127, c_out[-1]*127, j_out[-1]*127,  (c_out[-1]-j_out[-1])*127 , e_out[-1]]
        plots = len(graph_items)
        cmap = ['jet', 'jet', 'jet', 'jet', 'bwr']
        vmin = [0,0,0,-10,-1]
        vmax = [127,127,127,10,1]
        names = ["Actual", "Classical", "Jazz", "Difference", "Encoded"]


        for i in xrange(0, plots):
            fig.add_subplot(1,plots,i+1)
            plt.imshow(graph_items[i], vmin=vmin[i], vmax=vmax[i], cmap=cmap[i], aspect='auto')

            a = plt.colorbar(aspect=80)
            a.ax.tick_params(labelsize=7)
            ax = plt.gca()
            ax.xaxis.tick_top()

            if i == 0:
                ax.set_ylabel('Time Step')
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='both', labelsize=7)
            fig.subplots_adjust(top=0.85)
            ax.set_title(names[i], y=1.09)
            # plt.tight_layout()

        if self.one_hot:
            plt.xlim(0,88)
        else:
            plt.xlim(0,128)

        #Don't show the figure and save it
        if not path:
            out_png = os.path.join(self.dirs['png_path'], filename.split('.')[0] + "-e%d" % (epoch)+".png")
            plt.savefig(out_png, bbox_inches='tight')
            plt.close(fig)
        else:
            # out_png = os.path.join(self.dirs['png_path'], filename.split('.')[0] + "-e%d" % (epoch)+".png")
            # plt.savefig(out_png, bbox_inches='tight')
            # plt.close(fig)
            plt.show()
            plt.close(fig)
