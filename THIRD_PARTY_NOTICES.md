# Third-Party Notices

This repository contains the `whisperx-local` application code. It does not
bundle Python dependencies, FFmpeg, speech models, or speaker-diarization
models. Those components are installed or downloaded separately at runtime and
remain subject to their own licenses and terms.

## Runtime software

The pinned direct dependencies and the principal runtimes used by this project
are:

| Component                      | License      | Source                                         |
| ------------------------------ | ------------ | ---------------------------------------------- |
| WhisperX 3.8.6                 | BSD-2-Clause | https://github.com/m-bain/WhisperX             |
| Rich                           | MIT          | https://github.com/Textualize/rich             |
| CTranslate2                    | MIT          | https://github.com/OpenNMT/CTranslate2         |
| faster-whisper                 | MIT          | https://github.com/SYSTRAN/faster-whisper      |
| pyannote.audio                 | MIT          | https://github.com/pyannote/pyannote-audio     |
| PyTorch                        | BSD-3-Clause | https://github.com/pytorch/pytorch             |
| torchaudio                     | BSD-2-Clause | https://github.com/pytorch/audio               |
| Transformers                   | Apache-2.0   | https://github.com/huggingface/transformers    |
| Hugging Face Hub               | Apache-2.0   | https://github.com/huggingface/huggingface_hub |
| NLTK                           | Apache-2.0   | https://github.com/nltk/nltk                   |
| whisper.cpp (optional backend) | MIT          | https://github.com/ggml-org/whisper.cpp        |

Transitive dependencies may add their own notices. When redistributing a
packaged environment or binary, generate and ship a complete dependency notice
set from the exact lock or wheel set being distributed.

FFmpeg is an external executable discovered on `PATH`; this repository does
not distribute it. The license of an FFmpeg build depends on how that build was
configured, so follow the license and notice files supplied by the package
manager or distributor you use.

## Downloaded models

Models are not included in this repository. Before downloading or redistributing
one, review its current model card and access terms:

| Model or model family                     | Published terms                | Source                                                           |
| ----------------------------------------- | ------------------------------ | ---------------------------------------------------------------- |
| OpenAI Whisper models                     | MIT                            | https://huggingface.co/openai                                    |
| Systran faster-whisper-large-v3           | MIT                            | https://huggingface.co/Systran/faster-whisper-large-v3           |
| deepdml faster-whisper-large-v3-turbo-ct2 | MIT                            | https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2 |
| whisper.cpp GGML models                   | Review the selected model card | https://huggingface.co/ggerganov/whisper.cpp                     |
| whisper.cpp VAD models                    | MIT                            | https://huggingface.co/ggml-org/whisper-vad                      |
| pyannote speaker-diarization-community-1  | CC-BY-4.0                      | https://huggingface.co/pyannote/speaker-diarization-community-1  |

The pyannote model requires accepting the access conditions on Hugging Face.
The application uses a user-provided Hugging Face token and does not bypass
those conditions. CC-BY-4.0 attribution requirements apply to any permitted
redistribution or public use of that model and its outputs where applicable.

This notice is a project inventory, not a replacement for the upstream license
texts or model cards. Verify the terms again before shipping bundled models,
containers, installers, or a frozen executable.
