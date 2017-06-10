from collections import defaultdict
import copy
from math import log, floor, ceil
import pprint
import matplotlib.pyplot as plt
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import random

DEBUG = False

# The MIDI pitches we use.
PITCHES = xrange(21,109,1)
OFFSET = 109-21
PITCHES_MAP = { p : i for i, p in enumerate(PITCHES) }
print len(PITCHES)

def nearest_pow2(x):
    '''Normalize input to nearest power of 2, or midpoints between
    consecutive powers of two. Round down when halfway between two
    possibilities.'''

    low = 2**int(floor(log(x, 2)))
    high = 2**int(ceil(log(x, 2)))
    mid = (low + high) / 2

    if x < mid:
        high = mid
    else:
        low = mid
    if high - x < x - low:
        nearest = high
    else:
        nearest = low
    return nearest

def midi_to_array_one_hot(mid, quantization):
    '''Return array representation of a 4/4 time signature, MIDI object.

    Normalize the number of time steps in track to a power of 2. Then
    construct a T x N*2 array A (T = number of time steps, N = number of
    MIDI note numbers) where [A(t,n), A(t, n+1)] is the state of the note number
    at time step t.

    Arguments:
    mid -- MIDI object with a 4/4 time signature
    quantization -- The note duration, represented as 1/2**quantization.'''

    time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
    assert len(time_sig_msgs) == 1, 'No time signature found'
    time_sig = time_sig_msgs[0]
    assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'

    # Quantize the notes to a grid of time steps.
    mid = quantize(mid, quantization=quantization)

    # Convert the note timing and velocity to an array.
    _, track = get_note_track(mid)
    ticks_per_quarter = mid.ticks_per_beat
    time_msgs = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_msgs])

    track_len_ticks = cum_times[-1]
    if DEBUG:
        print 'Track len in ticks:', track_len_ticks
    notes = [
        (time * (2**quantization/4) / (ticks_per_quarter), msg.type, msg.note, msg.velocity)
        for (time, msg) in zip(cum_times, time_msgs)
        if msg.type == 'note_on' or msg.type == 'note_off']

    num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))
    normalized_num_steps = nearest_pow2(num_steps)
    notes.sort(key=lambda (position, note_type, note_num, velocity):(position,-velocity))

    if DEBUG:
        # pp = pprint.PrettyPrinter()
        print num_steps
        print normalized_num_steps
        # pp.pprint(notes)

    midi_array = np.zeros((normalized_num_steps, len(PITCHES)*2))
    velocity_array = np.zeros((normalized_num_steps, len(PITCHES)))
    open_msgs = defaultdict(list)

    for (position, note_type, note_num, velocity) in notes:
        if position == normalized_num_steps:
            # print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
            position = normalized_num_steps - 1
            # continue

        if position > normalized_num_steps:
            # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
            continue

        if note_type == "note_on" and velocity > 0:
            open_msgs[note_num].append((position, note_type, note_num, velocity))
            midi_array[position, 2*PITCHES_MAP[note_num]] = 1
            midi_array[position, 2*PITCHES_MAP[note_num]+1] = 1
            velocity_array[position, PITCHES_MAP[note_num]] = velocity
        elif note_type == 'note_off' or (note_type == 'note_on' and velocity == 0):

            note_on_open_msgs = open_msgs[note_num]

            if len(note_on_open_msgs) == 0:
                print 'Bad MIDI, Note has no end time.'
                return

            stack_pos, _, _, vel = note_on_open_msgs[0]
            open_msgs[note_num] = note_on_open_msgs[1:]
            current_pos = position
            while current_pos > stack_pos:
                # if midi_array[position, PITCHES_MAP[note_num]] != 1:
                midi_array[current_pos, 2*PITCHES_MAP[note_num]] = 0
                midi_array[current_pos, 2*PITCHES_MAP[note_num]+1] = 1
                velocity_array[current_pos, PITCHES_MAP[note_num]] = vel
                current_pos -= 1

    for (position, note_type, note_num, velocity) in notes:
        if position == normalized_num_steps:
            print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
            position = normalized_num_steps - 1
            # continue

        if position > normalized_num_steps:
            # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
            continue
        if note_type == "note_on" and velocity > 0:
            open_msgs[note_num].append((position, note_type, note_num, velocity))
            midi_array[position, 2*PITCHES_MAP[note_num]] = 1
            midi_array[position, 2*PITCHES_MAP[note_num]+1] = 1
            velocity_array[position, PITCHES_MAP[note_num]] = velocity

    assert len(midi_array) == len(velocity_array)
    return midi_array, velocity_array

def print_array(mid, array, quantization=4):
    '''Print a binary array representing midi notes.'''
    bar = 1
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_slice = ticks_per_beat/2**quantization

    bars = [x*ticks_per_slice % ticks_per_beat for x in xrange(0,len(array))]

    # print ticks_per_beat, ticks_per_slice
    res = ''
    for i, slice in enumerate(array):
        for pitch in slice:
            if pitch > 0:
                res += str(int(pitch))
            else:
                res += '-'
        if bars[i]== 0:
            res += str(bar)
            bar +=1
        res += '\n'
    # Take out the last newline
    print res[:-1]

def get_note_track(mid):
    '''Given a MIDI object, return the first track with note events.'''

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on':
                return i, track
    raise ValueError(
        'MIDI object does not contain any tracks with note messages.')

def quantize_tick(tick, ticks_per_quarter, quantization):
    '''Quantize the timestamp or tick.

    Arguments:
    tick -- An integer timestamp
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as 1/2**quantization
    '''
    assert (ticks_per_quarter * 4) % 2 ** quantization == 0, \
        'Quantization too fine. Ticks per quantum must be an integer.'
    ticks_per_quantum = (ticks_per_quarter * 4) / float(2 ** quantization)
    quantized_ticks = int(
        round(tick / float(ticks_per_quantum)) * ticks_per_quantum)
    return quantized_ticks

def unquantize(mid, style_mid):
    unquantized_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    orig_note_track_idx, orig_note_track = get_note_track(mid)
    style_note_track_idx, style_note_track = get_note_track(style_mid)

    note_track = unquantize_track(orig_note_track, style_note_track)
    unquantized_mid.tracks[orig_note_track_idx] = note_track

    return unquantized_mid

def unquantize_track(orig_track, style_track):
    '''Returns the unquantised orig_track with encoded velocities from the style_track.

    Arguments:
    orig_track -- Non-quantised MIDI object
    style_track -- Quantised and stylised MIDI object '''

    first_note_msg_idx = None

    for i, msg in enumerate(orig_track):
        if msg.type == 'note_on':
            orig_first_note_msg_idx = i
            break

    for i, msg in enumerate(style_track):
        if msg.type == 'note_on':
            style_first_note_msg_idx = i
            break

    orig_cum_msgs = zip(
        np.cumsum([msg.time for msg in orig_track[orig_first_note_msg_idx:]]),
        [msg for msg in orig_track[orig_first_note_msg_idx:]])

    style_cum_msgs = zip(
        np.cumsum([msg.time for msg in style_track[style_first_note_msg_idx:]]),
        [msg for msg in style_track[style_first_note_msg_idx:]])

    orig_cum_msgs.sort(key=lambda (cum_time, msg): cum_time)
    style_cum_msgs.sort(key=lambda (cum_time, msg): cum_time)

    open_msgs = defaultdict(list)

    for cum_time, msg in orig_cum_msgs:
        if msg.type == 'note_on' and msg.velocity > 0:
            open_msgs[msg.note].append((cum_time,msg))

    for i, (cum_time, msg) in enumerate(style_cum_msgs):
         if msg.type == 'note_on' and msg.velocity > 0:
            note_on_open_msgs = open_msgs[msg.note]
            note_on_cum_time, note_on_msg = note_on_open_msgs[0]
            note_on_msg.velocity = msg.velocity
            open_msgs[msg.note] = note_on_open_msgs[1:]

    return orig_track

def quantize(mid, quantization=5):
    '''Return a midi object whose notes are quantized to
    1/2**quantization notes.

    Arguments:
    mid -- MIDI object
    quantization -- The note duration, represented as
      1/2**quantization.'''

    quantized_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    note_track_idx, note_track = get_note_track(mid)
    new_track = quantize_track( note_track, mid.ticks_per_beat, quantization)
    if new_track == None:
        return None
    quantized_mid.tracks[note_track_idx] = new_track
    return quantized_mid

def quantize_track(track, ticks_per_quarter, quantization):
    '''Return the differential time stamps of the note_on, note_off, and
    end_of_track events, in order of appearance, with the note_on events
    quantized to the grid given by the quantization.

    Arguments:
    track -- MIDI track containing note event and other messages
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as
      1/2**quantization.'''

    pp = pprint.PrettyPrinter()

    # Message timestamps are represented as differences between
    # consecutive events. Annotate messages with cumulative timestamps.

    # Assume the following structure:
    # [header meta messages] [note messages] [end_of_track message]
    first_note_msg_idx = None
    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    cum_msgs = zip(
        np.cumsum([msg.time for msg in track[first_note_msg_idx:]]),
        [msg for msg in track[first_note_msg_idx:]])
    end_of_track_cum_time = cum_msgs[-1][0]

    quantized_track = MidiTrack()
    quantized_track.extend(track[:first_note_msg_idx])
    # Keep track of note_on events that have not had an off event yet.
    # note number -> message
    open_msgs = defaultdict(list)
    quantized_msgs = []
    for cum_time, msg in cum_msgs:
        if DEBUG:
            print 'Message:', msg
            print 'Open messages:'
            pp.pprint(open_msgs)
        if msg.type == 'note_on' and msg.velocity > 0:
            # Store until note off event. Note that there can be
            # several note events for the same note. Subsequent
            # note_off events will be associated with these note_on
            # events in FIFO fashion.
            open_msgs[msg.note].append((cum_time, msg))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # assert msg.note in open_msgs, \
            #     'Bad MIDI. Cannot have note off event before note on event'

            if msg.note not in open_msgs:
                 print 'Bad MIDI. Cannot have note off event before note on event'
                 return

            note_on_open_msgs = open_msgs[msg.note]

            if len(note_on_open_msgs) == 0:
                print 'Bad MIDI, Note has no end time.'
                return

            # assert len(note_on_open_msgs) > 0, 'Bad MIDI, Note has no end time.'

            note_on_cum_time, note_on_msg = note_on_open_msgs[0]
            open_msgs[msg.note] = note_on_open_msgs[1:]

            # Quantized note_on time
            quantized_note_on_cum_time = quantize_tick(
                note_on_cum_time, ticks_per_quarter, quantization)

            # The cumulative time of note_off is the quantized
            # cumulative time of note_on plus the orginal difference
            # of the unquantized cumulative times.
            quantized_note_off_cum_time = quantized_note_on_cum_time + (cum_time - note_on_cum_time)
            quantized_msgs.append((min(end_of_track_cum_time, quantized_note_on_cum_time), note_on_msg))
            quantized_msgs.append((min(end_of_track_cum_time, quantized_note_off_cum_time), msg))

            if DEBUG:
                print 'Appended', quantized_msgs[-2:]
        elif msg.type == 'end_of_track':
            quantized_msgs.append((cum_time, msg))

        if DEBUG:
            print '\n'

    # Now, sort the quantized messages by (cumulative time,
    # note_type), making sure that note_on events come before note_off
    # events when two event have the same cumulative time. Compute
    # differential times and construct the quantized track messages.
    quantized_msgs.sort(
        key=lambda (cum_time, msg): cum_time
        if (msg.type=='note_on' and msg.velocity > 0) else cum_time + 0.5)

    diff_times = [quantized_msgs[0][0]] + list(
        np.diff([ msg[0] for msg in quantized_msgs ]))
    for diff_time, (cum_time, msg) in zip(diff_times, quantized_msgs):
        quantized_track.append(msg.copy(time=diff_time))
    if DEBUG:
        print 'Quantized messages:'
        pp.pprint(quantized_msgs)
        pp.pprint(diff_times)
    return quantized_track

def stylify(mid, velocity_array, quantization):
    style_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    note_track_idx, note_track = get_note_track(mid)
    new_track = stylify_track(mid, velocity_array, quantization)
    style_mid.tracks[note_track_idx] = new_track
    return style_mid

# def midi_to_array(mid, quantization):
#     '''Return array representation of a 4/4 time signature, MIDI object.
#
#     Normalize the number of time steps in track to a power of 2. Then
#     construct a T x N array A (T = number of time steps, N = number of
#     MIDI note numbers) where A(t,n) is the velocity of the note number
#     n at time step t if the note is active, and 0 if it is not.
#
#     Arguments:
#     mid -- MIDI object with a 4/4 time signature
#     quantization -- The note duration, represented as 1/2**quantization.'''
#
#     time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
#     assert len(time_sig_msgs) == 1, 'No time signature found'
#     time_sig = time_sig_msgs[0]
#     assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'
#
#     # Quantize the notes to a grid of time steps.
#     mid = quantize(mid, quantization=quantization)
#
#     # Convert the note timing and velocity to an array.
#     _, track = get_note_track(mid)
#     ticks_per_quarter = mid.ticks_per_beat
#     time_msgs = [msg for msg in track if hasattr(msg, 'time')]
#     cum_times = np.cumsum([msg.time for msg in time_msgs])
#
#     track_len_ticks = cum_times[-1]
#     if DEBUG:
#         print 'Track len in ticks:', track_len_ticks
#     notes = [
#         (time * (2**quantization/4) / (ticks_per_quarter), msg.type, msg.note, msg.velocity)
#         for (time, msg) in zip(cum_times, time_msgs)
#         if msg.type == 'note_on' or msg.type == 'note_off']
#
#     num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))
#     normalized_num_steps = nearest_pow2(num_steps)
#     notes.sort(key=lambda (position, note_type, note_num, velocity):(position,-velocity))
#
#     if DEBUG:
#         # pp = pprint.PrettyPrinter()
#         print num_steps
#         print normalized_num_steps
#         # pp.pprint(notes)
#
#     midi_array = np.zeros((normalized_num_steps, len(PITCHES)))
#     velocity_array = np.zeros((normalized_num_steps, len(PITCHES)))
#     open_msgs = defaultdict(list)
#
#     for (position, note_type, note_num, velocity) in notes:
#         if position == normalized_num_steps:
#             # print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
#             position = normalized_num_steps - 1
#             # continue
#
#         if position > normalized_num_steps:
#             # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
#             continue
#
#         if note_type == "note_on" and velocity > 0:
#             open_msgs[note_num].append((position, note_type, note_num, velocity))
#             midi_array[position, PITCHES_MAP[note_num]] = 1
#             velocity_array[position, PITCHES_MAP[note_num]] = velocity
#         elif note_type == 'note_off' or (note_type == 'note_on' and velocity == 0):
#
#             note_on_open_msgs = open_msgs[note_num]
#
#             if len(note_on_open_msgs) == 0:
#                 print 'Bad MIDI, Note has no end time.'
#                 return
#
#             stack_pos, _, _, vel = note_on_open_msgs[0]
#             open_msgs[note_num] = note_on_open_msgs[1:]
#             current_pos = position
#             while current_pos > stack_pos:
#                 # if midi_array[position, PITCHES_MAP[note_num]] != 1:
#                 midi_array[current_pos, PITCHES_MAP[note_num]] = 2
#                 velocity_array[current_pos, PITCHES_MAP[note_num]] = vel
#                 current_pos -= 1
#
#     for (position, note_type, note_num, velocity) in notes:
#         if position == normalized_num_steps:
#             print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
#             position = normalized_num_steps - 1
#             # continue
#
#         if position > normalized_num_steps:
#             # print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
#             continue
#         if note_type == "note_on" and velocity > 0:
#             open_msgs[note_num].append((position, note_type, note_num, velocity))
#             midi_array[position, PITCHES_MAP[note_num]] = 1
#             velocity_array[position, PITCHES_MAP[note_num]] = velocity
#
#     return midi_array, velocity_array

def stylify_track(mid, velocity_array, quantization):

    _, track = get_note_track(mid)
    # first_note_msg_idx = None
    #
    # for i, msg in enumerate(track):
    #     if msg.type == 'note_on':
    #         first_note_msg_idx = i
    #         break

    ticks_per_quarter = mid.ticks_per_beat

    time_msgs = [msg for msg in track if hasattr(msg, 'time')]

    cum_times = np.cumsum([msg.time for msg in time_msgs])
    track_len_ticks = cum_times[-1]

    num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))
    normalized_num_steps = nearest_pow2(num_steps)
    # notes.sort(key=lambda (position, note_type, note_num, velocity):(position,-velocity))

    notes = [
        (time * (2**quantization/4) / (ticks_per_quarter), msg.type, msg.note, msg.velocity)
        for (time, msg) in zip(cum_times, time_msgs)
        if msg.type == 'note_on' or msg.type == 'note_off']

    cum_index = 0
    for i, time_msg in enumerate(track):
        if hasattr(time_msg, 'time'):
            if time_msg.type == 'note_on' or time_msg.type == 'note_off':
                if time_msg.velocity > 0:
                    pos = cum_times[cum_index] * (2**quantization/4) / (ticks_per_quarter)
                    if pos == normalized_num_steps:
                        pos = pos - 1
                    if pos > normalized_num_steps:
                        continue
                    vel = velocity_array[pos, PITCHES_MAP[time_msg.note]]
                    vel = vel*127
                    # print vel
                    vel = max(vel,1)
                    track[i].velocity = int(round(vel))
            cum_index += 1

    return track

def scrub(mid, velocity=10, random=False):
    '''Returns a midi object with one global velocity.

    Sets all velocities to a contant.

    Arguments:
    mid -- MIDI object with a 4/4 time signature
    velocity -- The global velocity'''
    scrubbed_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    note_track_idx, note_track = get_note_track(mid)
    if random:
        new_track = scrub_track_random(note_track)
    else:
        new_track = scrub_track(note_track,velocity=10)
    scrubbed_mid.tracks[note_track_idx] = new_track
    return scrubbed_mid

def scrub_track_random(track):

    first_note_msg_idx = None

    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    note_msgs = track[first_note_msg_idx:]

    for msg in note_msgs:
         if msg.type == 'note_on' and msg.velocity > 0:
             msg.velocity = random.randint(0,127)

    return track

def velocity_range(mid):
    '''Returns a count of velocities.

    Counts the range of velocities in a midi object.

    Arguments:
    mid -- MIDI object with a 4/4 time signature'''

    _, track = get_note_track(mid)
    first_note_msg_idx = None

    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break
    velocities = defaultdict(lambda:0)
    note_msgs = track[first_note_msg_idx:]
    for msg in note_msgs:
         if msg.type == 'note_on' and msg.velocity > 0:
             velocities[str(msg.velocity)] += 1
    dynamics = len(velocities.keys())
    # print velocities
    if dynamics > 1:
        return dynamics
    else:
        return 0

def scrub_track(track, velocity):
    first_note_msg_idx = None

    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    note_msgs = track[first_note_msg_idx:]

    for msg in note_msgs:
         if msg.type == 'note_on' and msg.velocity > 0:
             msg.velocity = 10

    return track
