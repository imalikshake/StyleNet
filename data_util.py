import numpy as np

class BatchGenerator(object):
    '''Generator for returning shuffled batches.

    data_x -- list of input matrices
    data_y -- list of output matrices
    batch_size -- size of batch
    input_size -- input width
    output_size -- output width
    mini -- create subsequences for truncating backprop
    mini_len -- truncated backprop window'''

    def __init__(self, data_x, data_y, batch_size, input_size, output_size, mini=True, mini_len=200):
        self.input_size = input_size
        self.output_size = output_size
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.batch_count = len(range(0, len(self.data_x), self.batch_size))
        self.batch_length = None
        self.mini = mini
        self.mini_len = mini_len


    def batch(self):
        while True:
            idxs = np.arange(0, len(self.data_x))
            np.random.shuffle(idxs)
            # np.random.shuffle(idxs)
            shuff_x = []
            shuff_y = []
            for i in idxs:
                shuff_x.append(self.data_x[i])
                shuff_y.append(self.data_y[i])

            for batch_idx in range(0, len(self.data_x), self.batch_size):
                input_batch = []
                output_batch = []
                for j in xrange(batch_idx, min(batch_idx+self.batch_size,len(self.data_x)), 1):
                    input_batch.append(shuff_x[j])
                    output_batch.append(shuff_y[j])
                input_batch, output_batch, seq_len = self.pad(input_batch, output_batch)
                yield input_batch, output_batch, seq_len


    def pad(self, sequence_X, sequence_Y):
        current_batch = len(sequence_X)
        padding_X = [0]*self.input_size
        padding_Y = [0]*self.output_size

        lens = [sequence_X[i].shape[0] for i in range(len(sequence_X))]
        # lens2 = [sequence_Y[i].shape[0] for i in range(len(sequence_Y))]
        #
        max_lens = max(lens)
        # max_lens2 = max(lens2)
        #
        # assert max_lens == max_lens2
        # print(max_lens)
        for i, x in enumerate(lens):
            length = x
            a = list(sequence_X[i])
            b = list(sequence_Y[i])
            while length < max_lens:
                a.append(padding_X)
                b.append(padding_Y)
                length+=1

            if self.mini:
                while length % self.mini_len != 0:
                    a.append(padding_X)
                    b.append(padding_Y)
                    length+=1

            sequence_X[i] = np.array(a)
            sequence_Y[i] = np.array(b)
            # for x in minis:
            #     mini_X.append(np.array(a[x:min(x+self.mini,x)]))
            #     mini_Y.append(np.array(b[x:min(x+self.mini,x)]))
            # print sequence_X[i].shape
            # print sequence_Y[i].shape

        # assert all(x.shape == (max_lens, self.input_size) for x in sequence_X)
        # assert all(y.shape == (max_lens, self.output_size) for y in sequence_Y)

        sequence_X = np.vstack([np.expand_dims(x, 1) for x in sequence_X])
        sequence_Y = np.vstack([np.expand_dims(y, 1) for y in sequence_Y])

        if not self.mini:
            mini_batches = 1
            max_lens = max(lens)
        else:
            mini_batches = length/self.mini_len
            max_lens = self.mini_len

        sequence_X = np.reshape(sequence_X, [current_batch*mini_batches, max_lens, self.input_size])
        sequence_Y = np.reshape(sequence_Y, [current_batch*mini_batches, max_lens, self.output_size])

        return sequence_X, sequence_Y, max_lens
