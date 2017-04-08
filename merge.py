from __future__ import print_function
import numpy as np
import h5py
from data_util import BatchGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from midi_util import *
from file_util import *

import os
import sys
from sklearn.preprocessing import MinMaxScaler


orignal_path = "/Users/Iman/research/programs/style/runs/first_run/midis/original"
quant_path = "/Users/Iman/research/programs/style/runs/first_run/midis/quantized"
pred_path = "/Users/Iman/research/programs/style/runs/first_run/velocities"


mids = []
quant_mids = []
velocities = []
filenames = []

path_prefix, path_suffix = os.path.split(orignal_path)

# Handle case where a trailing / requires two splits.
if len(path_suffix) == 0:
    path_prefix, path_suffix = os.path.split(path_prefix)
#
# for root, dirs, files in os.walk(orignal_path):
#     for file in files:
#         if file.split('.')[-1] == 'mid':
#             midi_path = os.path.join(root, file)
#             mid_file = MidiFile(midi_path)
#             filenames.append(file)
#             mids.append(mid_file)
#
# for root, dirs, files in os.walk(quant_path):
#     for file in files:
#         if file.split('.')[-1] == 'mid':
#             midi_path = os.path.join(root, file)
#             mid_file = MidiFile(midi_path)
#             quant_mids.append(mid_file)

for root, dirs, files in os.walk(pred_path):
    for file in files:
        if file.split('.')[-1] == 'npy':
            vel_path = os.path.join(root, file)
            loaded = np.load(vel_path)
            # loaded=loaded[-1]
            # print(loaded[1])
            plt.figure()
            plt.imshow(loaded)
            plt.show()
            velocities.append(loaded)

#
#
# scrubbed_mids = []
# for file in quant_mids:
#     scrub_mid = scrub(file,10)
#     scrubbed_mids.append(scrub_mid)
#
# scrubbed_orig_mids = []
# for file in mids:
#     scrub_mid = scrub(file,10)
#     scrubbed_orig_mids.append(scrub_mid)
#
# scrubbed_out = os.path.join(path_prefix, "clean_midi")
#
# if not os.path.exists(scrubbed_out):
#     os.makedirs(scrubbed_out)
#
# for i, file in enumerate(scrubbed_orig_mids):
#     out_file = os.path.join(scrubbed_out, filenames[i])
#     file.save(out_file)
#
# styled_mids = []
# for i, file in enumerate(scrubbed_mids):
#     print(velocities[i][48])
#     print(velocities[i][47])
#     print(velocities[i][49])
#     style_mid = stylify(file, velocities[i])
#     styled_mids.append(style_mid)
#
# styled_out = os.path.join(path_prefix, "qfinal")
#
# if not os.path.exists(styled_out):
#     os.makedirs(styled_out)
#
# for i, file in enumerate(styled_mids):
#     out_file = os.path.join(styled_out, filenames[i])
#     file.save(out_file)
#
#
#
# final_mids = []
# for i, file in enumerate(scrubbed_orig_mids):
#     final_mid = unquantize(file, styled_mids[i])
#     final_mids.append(final_mid)
#
# final_out = os.path.join(path_prefix, "final")
# if not os.path.exists(final_out):
#     os.makedirs(final_out)
# for i, file in enumerate(final_mids):
#     out_file = os.path.join(final_out, filenames[i])
#     file.save(out_file)
