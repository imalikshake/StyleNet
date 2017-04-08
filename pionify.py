import pretty_midi
import matplotlib.pyplot as plt
import os
from mido import MidiFile
from midi_util import velocity_range, quantize
from random import shuffle

mid_path = '/Users/Iman/research/midi/jazz_format0'
# train_path = './midi/evaluation_train'
# test_path = './midi/evaluation_test'
out_path = '/Users/Iman/research/midi/jazz_final'

if not os.path.exists(out_path):
    os.makedirs(out_path)
# if not os.path.exists(test_path):
#     os.makedirs(test_path)

total = len(os.listdir(mid_path))
vels = []
idx = range(0,total)
shuffle(idx)
test = int(round(total*10/100))
idx = idx[:test]
# print idx

for i , filename in enumerate(os.listdir(mid_path)):
    print filename
    if filename.split('.')[-1] == 'mid' or filename.split('.')[-1] == 'MID' :
        # mid = MidiFile(os.path.join(mid_path, filename))
        # if i in idx:
        #     mid.save(os.path.join(test_path, filename))
        # else:
        #     mid.save(os.path.join(train_path, filename))

        print "%d / %d" % (i,total)
        try:
            # midi_data = pretty_midi.PrettyMIDI(os.path.join(mid_path, filename))
            mid = MidiFile(os.path.join(mid_path, filename))
        except (KeyError, IOError, IndexError, EOFError, ValueError):
            print "NAUGHTY"
            continue
        # #
        # time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
        # #
        # if len(time_sig_msgs) == 1:
        #     time_sig = time_sig_msgs[0]
        #     if not (time_sig.numerator == 4 and time_sig.denominator == 4):
        #         print '\tTime signature not 4/4. Skipping ...'
        #         continue
        # else:
        #     # print time_sig_msgs
        #     print '\tNo time signature. Skipping ...'
        #     continue
        #
        # mid_q = quantize(mid, 4)
        #
        # if not mid_q:
        #     print 'Invalid MIDI. Skipping...'
            # continue

        # midi_data.write(os.path.join(out_path, filename))
        # print velocity_range(mid)
        vels.append(velocity_range(mid))
        if velocity_range(mid) >= 20:
            mid.save(os.path.join(out_path, filename))
        # piano = [instrument for instrument in midi_data.instruments if instrument.program < 8 ]
        # piano = [instrument for instrument in piano if not instrument.is_drum ]
        # # #
        # if len(piano) > 0 and len(piano) < 3:
        #     for x in piano:
        #         x.program = 0
        #     midi_data.instruments = piano
        #     print filename + "\t" + str(len(piano))
        #     midi_data.write(os.path.join(out_path, filename))
        # # # mid.save(os.path.join(out_path, filename))
        # else:
        #     print '\tNO piano.'
        #     # print filename + "\t" + str(len(piano))
#
#
# print velocity_range
plt.hist(vels,bins=range(0,120))
plt.xlabel("Number of Velocites in a Song")
plt.ylabel("Frequency")
plt.show()
#
# # for i , filename in enumerate(os.listdir(out_path)):
# #     if filename.split('.')[-1] == 'mid':
# #         try:
# #             midi_data = pretty_midi.PrettyMIDI(os.path.join(out_path, filename))
# #         except (KeyError, IOError, IndexError):
# #             print "NAUGHTY"
# #             continue
# #         piano = [instrument for instrument in midi_data.instruments if not instrument.is_drum]
# #         if len(piano) > 0 and len(piano) < 3:
# #             midi_data.instruments = piano
# #             midi_data.write(os.path.join(out_path2, filename))
# #             print filename + "\t" + str(len(piano))
