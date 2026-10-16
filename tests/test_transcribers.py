import numpy as np

from amtflow.transcribers import MuScriptor


class _FakeModel:
    """Records the audio length it is given and emits one note past the end."""

    def transcribe(self, audio, **kw):
        from muscriptor.events import NoteEndEvent, NoteStartEvent
        wav, sr = audio
        self.seconds = wav.shape[-1] / sr
        inside = NoteStartEvent(pitch=60, start_time=1.0, index=0, instrument="acoustic_piano")
        late = NoteStartEvent(pitch=61, start_time=self.seconds + 0.5, index=1, instrument="acoustic_piano")
        yield NoteEndEvent(end_time=self.seconds + 2.0, start_event=inside)
        yield NoteEndEvent(end_time=self.seconds + 1.0, start_event=late)


def _muscriptor(min_tail=1.0):
    m = MuScriptor.__new__(MuScriptor)
    m.known, m.model = {"acoustic_piano"}, _FakeModel()
    m.batch_size, m.prelude_forcing, m.min_tail = 1, True, min_tail
    m.boundaries, m.cfg_coef = "fixed", 1.0
    return m


def test_short_tail_is_trimmed_and_late_notes_dropped():
    m = _muscriptor()
    notes = m(np.zeros(int(30.04 * 16000), np.float32), 16000, ["acoustic_piano"])
    assert abs(m.model.seconds - 30.0) < 1e-6                 # 0.04 s tail removed
    assert [n.pitch for n in notes] == [60] and notes[0].end == 30.0


def test_long_tail_is_kept():
    m = _muscriptor()
    m(np.zeros(int(32.0 * 16000), np.float32), 16000, None)
    assert abs(m.model.seconds - 32.0) < 1e-6
