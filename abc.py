import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
import sys

# rename 's/\.mid\././' *


file_wav = sys.argv[-1]

# j_mid_classical = "/Users/Iman/predict/original_jazz/classical/wavs"
# j_mid_jazz = "/Users/Iman/predict/original_jazz/jazz/wavs"
# c_mid_jazz = "/Users/Iman/predict/original_classical/jazz/wavs"
# c_mid_jazz = "/Users/Iman/predict/original_classical/classical/wavs"
# o_mid_jazz = "/Users/Iman/predict/original/jazz/wavs"
# o_mid_classical = "/Users/Iman/predict/original/classical/wavs"
#



song = AudioSegment.from_wav(file_wav)

chunk_length_ms = 10000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = file_wav+"_{0}".format(i) + ".wav"
    print "exporting", chunk_name
#         chunk.export(chunk_name, format="wav")
