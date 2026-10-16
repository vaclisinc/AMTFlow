import pretty_midi

from amtflow.multtipop import midi_notes


def test_yourmt3_singing_voice_track_reads_as_voice(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    for name, program in (("Singing Voice", 65), ("Alto Sax", 65)):
        ins = pretty_midi.Instrument(program=program, name=name)
        ins.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
        pm.instruments.append(ins)
    path = tmp_path / "x.mid"
    pm.write(str(path))
    assert sorted(n[3] for n in midi_notes(path)) == [52, 65]
