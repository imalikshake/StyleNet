#! /usr/bin/env ruby

# Script to generate a Format 0 midifile from a Format 1 source

require_relative 'midifile.rb'

if ARGV.length != 2
  puts "ARGS: <path-to-mid-input-dir> <path-to-mid-output-dir>'"
  exit
end
Dir.glob(File.join(ARGV[0],"*.mid")) do |item|
    puts item
    out = Midifile.new
    open(item) {|f|
    mr = Midifile.new f
    out.format = 0
    mr.each {|ev|
        ev.trkno = 0 if ev.trkno
        out.add(ev)
        }
    }
    Dir.mkdir(ARGV[1]) unless File.exists?(ARGV[1])
    out_name = File.basename(item, '.mid') + ".mid"
    # out_name = item
    out_name = File.join(ARGV[1],out_name)
    puts out_name
    open(out_name,"w") {|fw|
      out.to_stream(fw) if out.vet()
    }
end
