Vendored from PyPI `muq==0.1.0` (tencent-ailab/MuQ, MIT; see LICENSE). Model weights
(OpenMuQ/MuQ-large-msd-iter) are CC-BY-NC 4.0 (LICENSE_weights).

AMTFlow changes: removed `modules/flash_conformer.py` (unused, imports a removed
transformers path); `models/muq_model.py` builds a real `Wav2Vec2ConformerConfig` and uses
`recording_encoder.RecordingConformerEncoder` (hidden states as in transformers 4.x).
