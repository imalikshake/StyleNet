import tensorflow as tf
from data_util import BatchGenerator
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff()

class GenreLSTM(object):
    def __init__(self, dirs, bi=False, one_hot=True, input_size=176, output_size=88, num_layers=1, batch_count=16):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.batch_count = int(batch_count)
        self.dirs = dirs
        self.bi = bi
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

    def train(self, data, model=None, starting_epoch=0, clip_grad=True, epochs=1001, input_keep_prob=0.5, output_keep_prob=0.5, learning_rate=0.005, eval_epoch=10, save_epoch=1):
        self.mini_len = 150
        self.data = data

        if clip_grad:
            jazz_optimizer = self.clip_optimizer(learning_rate,self.jazz_loss)
            classical_optimizer = self.clip_optimizer(learning_rate,self.classical_loss)
        else:
            jazz_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.jazz_loss)
            classical_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.classical_loss)



        self.sess = tf.Session()

        if model:
            self.load(model)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.summary_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'train'), graph=self.sess.graph_def)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'test'), graph=self.sess.graph_def)

        classical_batcher = BatchGenerator(self.data["classical"]["X"], self.data["classical"]["Y"], self.batch_count, self.input_size, self.output_size)
        jazz_batcher = BatchGenerator(self.data["jazz"]["X"], self.data["jazz"]["Y"], self.batch_count, self.input_size, self.output_size)

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

                self.train_writer.add_summary(classical_summary, epoch*classical_batcher.batch_count + epoch)
                self.train_writer.add_summary(jazz_summary, epoch*jazz_batcher.batch_count + epoch)

            print("[*] Average Training MSE for Classical epoch %d: %.9f" % (epoch, classical_epoch_avg/classical_batcher.batch_count))
            print("[*] Average Training MSE for Jazz epoch %d: %.9f" % (epoch, jazz_epoch_avg/jazz_batcher.batch_count))


            if epoch % save_epoch == 0 :
                self.save(epoch)
            if epoch % eval_epoch == 0 :
                self.evaluate(epoch)

        print("[*] Training complete.")

    def load(self, model_name):
        print(" [*] Loading checkpoint...")
        self.saver = tf.train.Saver(max_to_keep=0)
        self.saver.restore(self.sess, os.path.join(self.dirs['model_path'], model_name))

    def save(self, epoch):
        print("[*] Saving checkpoint...")
        model_name =  "model-e%d.ckpt" % (epoch)
        self.saver = tf.train.Saver(max_to_keep=0)
        save_path = self.saver.save(self.sess, os.path.join(self.dirs['model_path'], model_name))
        print("[*] Model saved in file: %s" % save_path)

    def evaluate(self, epoch, pred_save=False):

        input_eval_path = os.path.join(self.dirs['eval_path'], "inputs")
        vel_eval_path = os.path.join(self.dirs['eval_path'], "velocities")

        c_input_eval_path = os.path.join(input_eval_path, "classical")
        c_vel_eval_path = os.path.join(vel_eval_path, "classical")

        j_input_eval_path = os.path.join(input_eval_path, "jazz")
        j_vel_eval_path = os.path.join(vel_eval_path, "jazz")

        input_folder = os.listdir(c_input_eval_path)
        file_count = len(input_folder)

        c_eval_loss = 0
        j_eval_loss = 0
        #CLASSICS
        for i, filename in enumerate(input_folder):
            if filename.split('.')[-1] == 'npy':

                vel_path = os.path.join(c_vel_eval_path, filename)
                input_path = os.path.join(c_input_eval_path, filename)

                true_vel = np.load(vel_path)/120
                loaded = np.load(input_path)

                if not self.one_hot:
                    loaded = loaded/2

                in_list = []
                out_list = []

                in_list.append(loaded)
                out_list.append(true_vel)

                input_len = len(loaded)
                input_len = [input_len] * len(in_list)

                c_error, c_out, j_out, e_out, summary = self.sess.run([self.classical_loss,
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

                c_eval_loss += c_error

                self.plot_evaluation(epoch, filename, c_out, j_out, e_out, out_list)
                # if pred_save:
                #     predicted = os.path.join(self.dirs['pred_path'], filename.split('.')[0] + "-e%d" % (epoch)+".npy")
                #     np.save(predicted, linear[-1])


        input_folder = os.listdir(j_input_eval_path)
        file_count = len(input_folder)

        #JAZZ
        for i, filename in enumerate(input_folder):
            if filename.split('.')[-1] == 'npy':

                vel_path = os.path.join(j_vel_eval_path, filename)
                input_path = os.path.join(j_input_eval_path, filename)

                true_vel = np.load(vel_path)/120
                loaded = np.load(input_path)

                if not self.one_hot:
                    loaded = loaded/2

                in_list = []
                out_list = []

                in_list.append(loaded)
                out_list.append(true_vel)

                input_len = len(loaded)
                input_len = [input_len] * len(in_list)

                j_error, j_out, c_out, e_out, summary = self.sess.run([self.jazz_loss,
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

                j_eval_loss += j_error

                self.plot_evaluation(epoch, filename, c_out, j_out, e_out, out_list)
                # if pred_save:
                #     predicted = os.path.join(self.dirs['pred_path'], filename.split('.')[0] + "-e%d" % (epoch)+".npy")
                #     np.save(predicted, linear[-1])

        print("[*] Classical Model evaluated.")
        print("[*] Average Test MSE for Classical epoch %d: %.9f" % (epoch, c_eval_loss/file_count))
        print("[*] Average Test MSE for Jazz epoch %d: %.9f" % (epoch, j_eval_loss/file_count))
        print("[*] Average Test MSE for Overal epoch %d: %.9f" % (epoch, j_eval_loss+c_eval_loss/2*file_count))

        self.test_writer.add_summary(summary, epoch)

    def plot_evaluation(self, epoch, filename, c_out, j_out, e_out, out_list):
        fig = plt.figure(figsize=(13,11), dpi=120)
        fig.suptitle(filename, fontsize=10, fontweight='bold')

        graph_items = [out_list[-1]*120, c_out[-1]*120, j_out[-1]*120,  (c_out[-1]-j_out[-1])*120 , e_out[-1]]
        plots = len(graph_items)
        cmap = ['jet', 'jet', 'jet', 'jet', 'bwr']
        vmin = [0,0,0,-10,-1]
        vmax = [120,120,120,10,1]
        names = ["Actual", "Classical", "Jazz", "Difference", "Encoded"]


        for i in xrange(0, plots):
            fig.add_subplot(1,plots,i+1)
            plt.imshow(graph_items[i], vmin=vmin[i], vmax=vmax[i], cmap=cmap[i], aspect='auto')

            a = plt.colorbar(aspect=80)
            a.ax.tick_params(labelsize=8)
            ax = plt.gca()
            ax.xaxis.tick_top()

            if i == 0:
                ax.set_ylabel('Time Step')
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='both', labelsize=8)
            fig.subplots_adjust(top=0.85)
            ax.set_title(names[i], y=1.09)
            # plt.tight_layout()

        if self.one_hot:
            plt.xlim(0,88)
        else:
            plt.xlim(0,128)

        out_png = os.path.join(self.dirs['png_path'], filename.split('.')[0] + "-e%d" % (epoch)+".png")
        plt.savefig(out_png, bbox_inches='tight')
        plt.close(fig)
