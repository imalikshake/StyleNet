# encoding: ASCII-8BIT
#############################################
### Midifile Input and Output Facilities ####
#
# Copyright (c) 2008-2014 by Pete Goodeve
#
# vers 2014/11/30 -- Ruby 1.9
#
#############################################


### Constant definitions etc. ###

HDR=0x00
END_OF_FILE=0x01

NOTE_OFF=0x80
NOTE_ON=0x90
POLY_TOUCH=0xa0
CONTROL_CHANGE=0xb0
PROGRAM_CHANGE=0xc0
CHANNEL_TOUCH=0xd0
PITCH_BEND=0xe0
SYSTEM=0xf8	# maybe... (not ever seen in midifile?)
SYSEX=0xf0
SYSEX_CONT=0xf7
META=0xff
TRK_START=0x100	## not a byte code!
TRK_END=0x1ff

EvType = {
	NOTE_OFF=>"NOTE OFF",
	NOTE_ON=>"NOTE ON",
	POLY_TOUCH=>"POLY TOUCH",
	CONTROL_CHANGE=>"CONTROL CHANGE",
	PROGRAM_CHANGE=>"PROGRAM CHANGE",
	CHANNEL_TOUCH=>"CHANNEL TOUCH",
	PITCH_BEND=>"PITCHBEND",
	SYSEX=>"SYSEX",
	SYSEX_CONT=>"SYSEX CONTINUATION",
	META=>"META",
	TRK_START=>"TRACK START",
	TRK_END=>"TRACK END",
	HDR=>"FILE HEADER",
	END_OF_FILE=>"FILE END"
}

## Meta Event types:

SEQ_NUM=0x00
TEXT=0x01
COPYRIGHT=0x02
TRACK_NAME=0x03
INSTR_NAME=0x04
LYRIC=0x05
MARKER=0x06
CUE_POINT=0x07
DEVICE_NAME=0x09
CHAN_PFX=0x20
MIDI_PORT=0x21
END_TRK=0x2f
TEMPO=0x51
SMPTE=0x54
TIME_SIG=0x58
KEY_SIG=0x59
SEQUENCER=0x7f

MetaType = {
	0x00=>'SEQ_NUM',	# 2-byte number
	0x01=>'TEXT',		# string
	0x02=>'COPYRIGHT',	# string
	0x03=>'TRACK_NAME',	# string
	0x04=>'INSTR_NAME',	# string
	0x05=>'LYRIC',		# string
	0x06=>'MARKER',		# string
	0x07=>'CUE_POINT',	# string
	0x09=>'DEVICE_NAME',# string
	0x20=>'CHAN_PFX',	# byte
	0x21=>'MIDI_PORT',	# byte
	0x2f=>'END_TRK',	# also in @code attr
	0x51=>'TEMPO',		# 3-byte usec/q-note
	0x54=>'SMPTE',		# 5-bytes: hr mn sc fr ff
	0x58=>'TIME_SIG',	# 4-bytes: nn dd cc bb
	0x59=>'KEY_SIG',	# 2-bytes: sf mi
	0x7f=>'SEQUENCER',	# Sequencer specific
}
##############################
# Bypassing Ruby 1.9's completely idiotic revamping of basics!!!

if String.instance_methods.include?(:getbyte) then

 class Array
	def nitems
		select{|x| x}.count
	end
 end

 class IO
	def getc
		getbyte
	end
 end

 class MString < String
    def [] i, *more
        if !i.is_a?(Integer) || !more.empty? then
             super(i, *more)
        else
            getbyte(i)
        end
    end
 end
 
else	#1.8.x or earlier...

 class MString < String
 end
 
end 

##############################


module MidifileOps

	# readByte() should be defined in class appropriately
	
	# Read a sixteen bit value.
	def read16
		val = (readByte() << 8) + readByte()
		val = val - 0x10000 if (val & 0x8000).nonzero?
		return val
		end

	# Read a 32-bit value.
	def read32
		val = (readByte() << 24) + (readByte() << 16) +
		  (readByte() << 8) + readByte()
		val = val - 0x100000000 if (val & 0x80000000).nonzero?
		return val
		end

	# Read a varlen value.
	def readVarlen
		c = readByte()
		val = 0
		p c if !c
		until !c || (c & 0x80).zero?
			val = (val  | (c & 0x7f)) << 7
			c = readByte()
			p c if !c
			end
		puts "Error: c was #{c} at #{@bytes_left}" if !c
		val |= c
		#puts "got VarLen #{val}"
		return val
		end

	# Generate bytes for a 16-bit value.
	def bytes2(val)
		val = (val - 0x10000) & 0xffff if val < 0
		s = '' << ((val >> 8) & 0xff)
		s << (val & 0xff)
	end

	# Generate bytes for a 32-bit value.
	def bytes4(val)
		val = (val - 0x100000000) & 0xffffffff if val < 0
		'' << ((val >> 24) & 0xff) << ((val >> 16) & 0xff) <<
			((val >> 8) & 0xff) << (val & 0xff)
	end

	# Generate bytes for a variable length value.
	def bytesVarlen(val)
		return "\000" if val.zero?
		buf = Array.new()
		s = '' << (val & 0x7f)
		while (val >>= 7) > 0
			s << ((val & 0x7f) | 0x80)
		end
		s.reverse
	end

end ### module MidifileOps


class MidiItem
	include MidifileOps
	def initialize(code, trkno=nil, time=nil, delta=0)
		@code = code
		@trkno = trkno
		@time = time
		@delta = delta
		end
	# May need to adjust any of these:
	attr_accessor :code, :time, :delta, :trkno
	attr_accessor :listindx	# used to restrict sorting's eagerness
	# Everything declared (readable) at this level
	# so we can check for existence without bombing...:
	attr_reader :format, :ntrks, :division
	attr_reader :chan, :data1, :data2, :running
	attr_reader :length, :data, :meta

	def to_s
		if @code == END_OF_FILE
			"EOF"
		else
			"#{@trkno ? @trkno : "--"}: #{@time? @time : "--"} #{@code}"
		end
	end
	def to_bytes
		''
	end
	# The 'channel' accessors handle user-range 1..16 rather than 0..15:
	def channel
		@chan ? @chan+1 : nil	# return 1..16 (to match 'gen...' methods)
	end
	def channel=(ch)
		@chan = ch - 1
	end
end ### MidiItem


class MidiHeader < MidiItem
	def initialize(format, ntrks, division)
		super(HDR)
		@format = format
		@ntrks = ntrks
		@division = division
		end
	def to_s
		"Format #{@format}: #{@ntrks} tracks -- division=#{@division}"
	end
	def to_bytes
		'MThd' << bytes4(6) << bytes2(@format) <<
			bytes2(@ntrks) << bytes2(@division) 
	end
end ### MidiHeader


class TrackHeader < MidiItem
	def initialize(trkno, length)
		super(TRK_START, trkno)
		@length = length
	end
	def to_s
		"#{@trkno ? @trkno : "--"}: -- TRACK_START length #{@length} bytes"
	end
	def to_bytes
		'MTrk' << bytes4(@length) 
	end
end ### TrackHeader


class MidiEvent < MidiItem
	# For Channel Events:
	attr_accessor :chan, :data1, :data2, :running
	# For System & Meta Events:
	attr_accessor :length, :data, :meta

	def initialize(code, trkno=nil, time=nil, delta=0)
		# An event may be specified as an array of the actual MIDI bytes
		# (not SysEx (yet))
		if code.is_a?(Array)
			if code[0] < 0xf0 then
				super(code[0] & 0xf0, trkno, time, delta)
				@chan = code[0] & 0x0f
			else
				super(code[0], trkno, time, delta)
			end
			@data1 = code[1] if code[1]
			@data2 = code[2] if code[2]
		else
			super(code, trkno, time, delta)
		end
	end

	def to_s
	  begin
		s = "#{@trkno}: #{@time} "
		if @chan
			s << "#{EvType[@code]} chan=#{@chan}"
			case code
			when NOTE_ON
			 s << " note #{@data1} velocity #{@data2}"
			when NOTE_OFF, POLY_TOUCH
			 s << " note #{@data1} value #{@data2}"
			when CONTROL_CHANGE
			 s << " controller #{@data1} value #{@data2}"
			when PROGRAM_CHANGE
			 s << " program #{@data1}"
			when PITCH_BEND, CHANNEL_TOUCH
			 s << " value #{@data1}"
			end
			 "#{@data1} #{@data2? @data2 : ''}"
			s << " [R]" if @running
		elsif @meta
			mcode = MetaType[@meta] 
			s << (mcode ? " #{mcode}" : "#{@code} 0x%x"%@meta)
			if (0x01..0x07) === @meta
				s << ' "' << @data << '"'
			elsif mcode == 'TEMPO'
				tempo = (data[0]*256 + data[1])*256 + data[2]
				s << " #{tempo} microsec/quarter-note"
			elsif @length && @length > 0
				s << " ["
				@data.each_byte {|b| s << " %d"%b}
				s << " ]"
			end
		end
		return s
	  rescue
	  	p self
	  	raise
	  end
	end
	
	def to_bytes
		## NOTE: it is assumed that structure is correct! (unneeded == nil)
		case @code
		when SYSTEM
			### Not handled (yet?)
			return nil
		when SYSEX, SYSEX_CONT, META, TRK_END
			command = @code
		else	## may need sanity check here...
			command = @code | @chan
		end
		s = '' << bytesVarlen(@delta)
		s << (command & 0xff) if !@running	# (must ensure is bytesize)
		if @length
			s << @meta if @meta
			s << bytesVarlen(@length) << @data
		elsif @code == TRK_END	# allow incomplete event struct
			s << 0x2f << 0x00
		elsif @code == PITCH_BEND
			val = @data1 + 8192
			s << (val & 0x7f)
			s << ((val >> 7) & 0x7f)
		elsif @data1
			s << @data1
			s << @data2 if @data2
		end
		return s
	end

	def to_midi
		# generate standard MIDI byte sequence
		return nil if @code >= META || @code < NOTE_OFF	#should never hit this?
		s = '' << (@code < SYSEX ? (@code | @chan) : @code)
		if @code == SYSEX || @code == SYSEX_CONT
			s << @data
		elsif @code == PITCH_BEND
			val = @data1 + 8192
			s << (val & 0x7f)
			s << ((val >> 7) & 0x7f)
		elsif @data1
			s << @data1
			s << @data2 if @data2
		end
		return s
	end
end ### MidiEvent


class MidiTrack
	include MidifileOps
	def initialize(trkno, src=nil)
		@src = src
		@trkno = trkno
		## ... and set up start and end etc...
		if src
			id, @trklen = src.read_spec()
			return nil if (id != 'MTrk')	## thought-holder... -- raise exception
		end
		@last_insert = 0
		@insert_time = 0
	end
	attr_reader :trkno, :trklen, :evlist

	# Read a single character from track
	def readByte
		byte = @src.instream.getc()
		@bytes_left -= 1 if byte
		return byte
	end

	def read_system_or_meta(code)
		length = 0
		data = MString.new
		case code
		when META
			# running status OK around this
			meta = readByte()
			length = readVarlen()
			code = TRK_END if meta == 0x2f
	 	when SYSEX, SYSEX_CONT
	 		@running = false	# just in case
		 	@command = nil	# maybe a litle protection from bad values...
			length = readVarlen()
	 	else
	 		@running = false	# just in case
	 		puts "unexpected system event #{code}"
			return nil	### TEMP for now...
		end
		ev = MidiEvent.new(code, @trkno, @elapsed, @delta)	# (temp)
		ev.meta = meta if meta
		ev.data = MString.new @src.instream.read(length) if length
		ev.length = length	# (excludes meta byte)
	 	@bytes_left -= length
		return ev
	end

	def read_event
		@delta = readVarlen() # Delta time
		@elapsed += @delta
		code = readByte()		# Read first byte
		if (code & 0x80).zero? # Running status?
			puts 'unexpected running status' if !@command || @command.zero?
			@running = true
		elsif code >= 0xf0
			return read_system_or_meta(code)
		else
			@command = code
			@running = false
		end
		##puts "Status %x, code=%x chan = %x"%[@command, (@command>>4)&7, @command & 0xf]
		ev = MidiEvent.new(@command&0xf0, @trkno, @elapsed, @delta)
		ev.chan = @command & 0xf
		ev.running = @running	# recorded for possible convenience
		ev.data1 = @running? code : readByte()
		case @command & 0xf0
		when NOTE_OFF..CONTROL_CHANGE
			ev.data2 =  readByte()
		when PROGRAM_CHANGE, CHANNEL_TOUCH
			#do nothing
		when PITCH_BEND
			msb = readByte()
			ev.data1 += msb*128 - 8192
		end
		return ev
	end

	# Read the track.
	def each
		c = c1 = type = needed = 0
		@sysex_continue = false	# True if last msg was unfinished
		@running = false		# True when running status used
		@command = nil		# (Possibly running) "status" byte

		@bytes_left = @trklen
		@elapsed = 0

		if @src
			yield read_event() while @bytes_left > 0
			return @elapsed
		elsif @evlist
			@evlist.each {|ev| yield ev}
		end
	end

  #######################
  ## Output section
  
	def add(ev)
		return true if not ev.is_a?(MidiEvent)	# so we can pass with no trouble
		return false if @src || (ev.trkno && ev.trkno != @trkno)
		if !@evlist || ev.code == TRK_START
			@evlist = []
			@trklen = 0
			return true if ev.code == TRK_START
		end
		if !ev.time ||
			 (@evlist.last && @evlist.last.time &&
			  ev.time >= @evlist.last.time) then
			@evlist << ev
		else
			indx = -1	# append if no insertion point found
			@last_insert = 0 if ev.time < @insert_time
			for i in (@last_insert...@evlist.length) do
				if @evlist[i].time && @evlist[i].time > ev.time then
					indx = i
					break
				end
			end
			@evlist.insert(indx, ev)
			@last_insert = indx
			@insert_time = ev.time
		end
		@trklen += ev.to_bytes.length
# 		puts "added event #{ev.code} making #{@trklen} bytes"
		return true
	end
	
	def empty?(end_alone_ok=nil)
		return true if !@evlist || @evlist.empty? 
		return true if @evlist.length == 1 &&
			@evlist.last.code == TRK_END && !end_alone_ok
		return false
	end

	def vet(use_running=true)
		return false if not @evlist
		time = 0
		# looks like three passes are needed:
		@evlist.each_with_index {|ev, indx|
			if !ev.time
				time = time + ev.delta if ev.delta
				ev.time = time
			elsif ev.time > time	# seems reasonable...
				time = ev.time
			end
			ev.listindx = indx	#prevent over-eager sorting
		}
		@evlist.sort!{|a,b|
			if a.time == b.time then
				res = a.listindx <=> b.listindx
			else
				res = a.time <=> b.time
			end
# 			puts "sorting #{a} against #{b} -- res #{res}"
			res
		}
		time = 0
		@trklen = 0
		curr_code = 0
		curr_chan = nil
		to_delete = []
		@evlist.each {|ev|
			if ev.code == TRK_END && ev != @evlist.last
 				to_delete << ev	## can't remove within the loop!!
 			else
				ev.delta = ev.time - time
				time = ev.time
				if use_running && ev.code == curr_code &&
				   ev.chan && ev.chan == curr_chan
					ev.running = true
				elsif ev.running
					ev.running = nil
				end
				curr_code = ev.code
				curr_chan = ev.chan
				@trklen += ev.to_bytes.length
 			end
		}
		to_delete.each {|ev| @evlist.delete(ev)} # trklen shouldn't include these
		if @evlist.last.code != TRK_END
			ev = MidiEvent.new(TRK_END, @trkno,  @evlist.last.time)
			ev.meta = 0x2f
			add(ev)
		end
		return true
	end

	def to_stream(stream)
		return false if !@evlist
		if @evlist.empty? || @evlist.last.code != TRK_END
			currtime = @evlist.last ? @evlist.last.time : 0
			ev = MidiEvent.new(TRK_END, @trkno, currtime)
			ev.meta = 0x2f
			add(ev)
		end
		stream << 'MTrk' << bytes4(@trklen)
		@evlist.each {|ev|
			stream << ev.to_bytes
		}
	end
	
end	### MidiTrack


class Midifile

	include MidifileOps

	def initialize(stream=nil)
		@instream = stream
	end

	attr_reader :instream, :tracks

	# Read a single character
	def readByte
		return @instream.getc()
	end

	# Read chunk spec
	def read_spec
		id = @instream.read(4)
		length = read32()
		return id,length
	end

	# Read the header
	def read_header_chunk
		id, size = read_spec()
		return nil if (id != 'MThd' || size != 6)	## crude for now...
		@format = read16()
		@ntrks = read16()
		@division = read16()
	end

	def each
		read_header_chunk() if @instream
		return nil if !@format
		@ntrks = @tracks.nitems if !@instream && @tracks
		yield MidiHeader.new(@format, @ntrks, @division)
		if @instream
			(0...@ntrks).each {|n|
				@elapsed = 0
				trk = MidiTrack.new(n, self)
				yield TrackHeader.new(n, trk.trklen)
				@elapsed = trk.each {|ev| yield ev}
			}
		elsif @tracks
			@tracks.compact.each {|trk|
				yield TrackHeader.new(trk.trkno, trk.trklen)
				trk.each {|ev| yield ev}
			}
		end
			yield MidiItem.new(END_OF_FILE)
	end

  #######################
  ## Output section
  
	def format=(format)
		# can't change existing value
		if !@format && (0...2) === format
			@format = format
		else
			return nil
		end
	end

	def division=(division)
		@division = division
	end

	def addTrack(n=nil)
		@tracks = @tracks || []
		if n then @tracks[n] = track = MidiTrack.new(n)
		else @tracks << track = MidiTrack.new(@tracks.length)
		end
		return track
	end

	def add(ev)
		if ev.code == HDR
			self.format=ev.format	# use method to protect against change!
 			@division=ev.division
			# number of tracks defined by array
		elsif ev.trkno	# ignore END_OF_FILE (etc?)
			addTrack(ev.trkno) if !@tracks || !@tracks[ev.trkno]
			@tracks[ev.trkno].add(ev)
		end
	end

	def vet(use_running=true)
		return false if !@tracks || !@format
		@tracks.compact.each {|trk|
			res = trk.vet(use_running)
			@tracks.delete(trk) if trk.empty?
			return false if not res
		}
		return true
	end

	def to_stream(stream)
		return false if !@tracks || !@format
		stream << 'MThd' << bytes4(6) << bytes2(@format) <<
			bytes2(@tracks.nitems) << bytes2(@division) 
		@tracks.compact.each {|trk|
			trk.to_stream(stream)
		}
		return true
	end

end ### Midifile

#####################################################################

### Utility Event Subclasses & Methods (to simplify event generation)

### Parameters in the following methods have these conventions:
###  By default, 'ticks' should be delta-ticks.
###  -- to use absolute elapsed ticks set MidiEvent.deltaTicks=false
###  WHen a channel is supplied, it should be in the range 1..16
###  (but 0 is also allowed as the first channel)
###  Track numbering however, starts at zero.
###  If there is neither a supplied track nor a default one, channel
###  events will get a track according to their channel; metaevents
###  will go to track 0.

class MidiEvent	# extension from main definition above
	# For User-created event convenience:
		@@defaultTrack = 0
		@@defaultChannel = 0
		@@useDeltas = true	# false for absolute ticks
	def MidiEvent.track=(trk)
		@@defaultTrack = trk	# can be nil for track=chan
	end
	def MidiEvent.channel=(chan)
		chan -= 1 if chan && (chan > 0)	# supplied range 1..16 
		@@defaultChannel = (chan || 0) & 0xF
	end
	def MidiEvent.deltaTicks=(usedeltas)
		@@useDeltas = usedeltas
	end
	def MidiEvent.track()
		@@defaultTrack
	end
	def MidiEvent.channel()
		@@defaultChannel+1	# returned in user notation (1..16)!
	end
	def MidiEvent.deltaTicks()
		@@useDeltas
	end
end # MidiEvent extension


class ChannelEvent < MidiEvent	# Convenience subclass
	def initialize(code, ticks=0, data1=nil, data2=nil, chan=nil, track=nil)
		chan -= 1 if chan && (chan > 0)	# supplied range 1..16 
		@chan = chan || @@defaultChannel
		track = track || @@defaultTrack
		track = track || @chan+1	# back to user convention for track number
		if @@useDeltas then
			super(code, track, nil, ticks)
		else
			super(code, track, ticks)
		end
		@data1 = data1
		@data2 = data2
	end
end ### ChannelEvent


def genNoteOff(ticks, note, vel=0, chan=nil, track=nil)
		ChannelEvent.new(NOTE_OFF, ticks, note, vel, chan, track)
end

def genNoteOn(ticks, note, vel, chan=nil, track=nil)
	ChannelEvent.new(NOTE_ON, ticks, note, vel, chan, track)
end

def genPolyTouch(ticks, note, pressure, chan=nil, track=nil)
	ChannelEvent.new(POLY_TOUCH, ticks, note, pressure, chan, track)
end

def genControlChange(ticks, controller, value, chan=nil, track=nil)
	# controller numbers start at 0!
	ChannelEvent.new(CONTROL_CHANGE, ticks, controller, value, chan, track)
end

def genProgramChange(ticks, program, chan=nil, track=nil)
	program -= 1 if program > 0	# numbers start at 1!
	ChannelEvent.new(PROGRAM_CHANGE, ticks, program, nil, chan, track)
end

def genChannelTouch(ticks, pressure, chan=nil, track=nil)
	ChannelEvent.new(CHANNEL_TOUCH, ticks, pressure, nil, chan, track)
end

def genPitchBend(ticks, bend, chan=nil, track=nil)	# Signed 14-bit value! (0 = no bend)
	ChannelEvent.new(PITCH_BEND, ticks, bend, nil, chan, track)	# kept in data1 only (until output)
end 


class MetaEvent < MidiEvent	# Convenience subclass
	def initialize(ticks, meta, length, data, track=nil)
		track = (track || @@defaultTrack) || 0	# use zero if no default
		if @@useDeltas then
			super(0xFF, track, nil, ticks)
		else
			super(0xFF, track, ticks)
		end
		@meta = meta
		@length = length
		if data.is_a?(String) then
			@data = MString.new data
		else	# assume array of bytes
			@data = MString.new 
			data.each {|b| @data << b}
		end
	end
end ### ChannelEvent


def genText(ticks, type, text, track=nil)
	meta = (TEXT..CUE_POINT) === type ? type : 1
	MetaEvent.new(ticks, meta, text.length, text, track)
end

def genTempo(ticks, micros=500000, track=nil)
	tempo = [(micros>>16) & 0xFF, (micros>>8) & 0xFF, micros & 0xFF]
	MetaEvent.new(ticks, 0x51, 3, tempo, track)
end

def genTimeSignature(ticks, numer, denom, metronome=24, notat32=8, track=nil)
## see the midifile spec!! -- except that denom is the actual one (e.g. 3/"8")
	dpow2 = 0
	dpow2 += 1 while 2**dpow2 < denom 
	MetaEvent.new(ticks, 0x58, 4, [numer, dpow2, metronome, notat32], track)
end

def genKeySignature(ticks, sharpsflats, minor=0, track=nil)
	MetaEvent.new(ticks, 0x59, 2, [sharpsflats, minor], track)
end

## For metaevents not covered by the above...:
def genMeta(ticks, meta, data, track=nil)
	MetaEvent.new(ticks, meta, data.length, data, track)
end

## 'data' should be either an array of byte values, or a string
def genSysEx(ticks, data, track=nil)
		track = (track || MidiEvent.track) || 0	# use zero if no default
		if MidiEvent.deltaTicks then
			ev = MidiEvent.new(0xF0, track, nil, ticks)
		else
			ev = MidiEvent.new(0xF0, track, ticks)
		end
		if data.is_a?(String) then
			ev.data = MString.new data
		else	# assume array of bytes
			ev.data = MString.new 
			data.each {|b| ev.data << b}
		end
		ev.data << 0xF7 if ev.data[-1] != 0xF7	# terminate as per protocol
		ev.length = ev.data.length
		return ev
end

