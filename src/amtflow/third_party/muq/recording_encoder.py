"""Wav2Vec2-Conformer encoder that returns hidden states with transformers 4.x semantics.

transformers 5 moved hidden-state capture to a hook on the top-level model, which
MuQ does not use. This returns N+1 tensors: the input to layer 1, the outputs of
layers 1..N-1, and the final LayerNorm'd output of layer N (AMTFlow addition).
"""

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
)


class RecordingConformerEncoder(Wav2Vec2ConformerEncoder):
    def forward(self, hidden_states, attention_mask=None, output_hidden_states=False, **kwargs):
        if attention_mask is not None:
            raise NotImplementedError("clips are encoded one at a time, unpadded")
        hidden_states = self.dropout(hidden_states)
        rel = self.embed_positions(hidden_states) if self.embed_positions is not None else None
        states = []
        for layer in self.layers:
            states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=None, relative_position_embeddings=rel)
        hidden_states = self.layer_norm(hidden_states)
        states.append(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=tuple(states))
