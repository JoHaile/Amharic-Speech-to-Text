# AI AGENT TASK: Install and Run Shook Tiny Amharic Speech-to-Text Locally

## 1. Objective

Download, configure, and run the **Shook Tiny Stage 2 Polish** Amharic speech recognition model on my Windows computer.

The goal is to have a working, completely offline Amharic speech-to-text system.

I want to provide an audio recording containing spoken Amharic and receive the transcription as Amharic text.

This is an installation and testing task, NOT a model-training task.

## 2. Hardware and Environment

My computer has:

- OS: Windows 10
- CPU: Intel Core i7, 8th Generation
- RAM: 16 GB
- Storage: SSD, 1 TB
- GPU: None
- CUDA: Not available

### Strict requirements

- CPU inference ONLY.
- No NVIDIA dependencies.
- No CUDA.
- No cloud processing.
- No external inference APIs.
- No paid services.
- No model training or fine-tuning.
- Everything must run locally.

Use CPU-compatible PyTorch and Hugging Face Transformers.

## 3. Model to Download

Primary model:

`b1n1yam/shook-tiny-amharic-stage2-polish`

Hugging Face:
https://huggingface.co/b1n1yam/shook-tiny-amharic-stage2-polish

Model information:

- Architecture: Whisper Tiny fine-tuned for Amharic.
- Parameters: 37.8 million.
- Reported evaluation WER: 20.7634%.
- Model format: Safetensors.
- License: Apache 2.0.

Use this exact checkpoint. Do not silently substitute another model.

## 4. Installation

Create a clean Python virtual environment inside a dedicated project directory.

For example:

`C:\Amharic-ASR`

Install the necessary dependencies:

- CPU-compatible PyTorch
- Transformers
- Hugging Face Hub
- SoundFile
- Librosa
- NumPy

Use compatible package versions.

IMPORTANT:
Install the CPU-only PyTorch build. Do not install CUDA-enabled PyTorch unnecessarily.

Download the complete model checkpoint and tokenizer/processor files from Hugging Face.

Store the downloaded model locally so that subsequent executions do not require an internet connection.

## 5. Implement the Transcription System

Create a simple Python application that:

1. Loads the downloaded model from the local directory.
2. Loads the audio processor.
3. Accepts an audio file as input.
4. Processes the audio using CPU inference.
5. Produces the recognized Amharic text.
6. Displays the transcription in the terminal.
7. Saves the transcription to a `.txt` file.

Use the appropriate Hugging Face automatic speech recognition interfaces.

Explicitly configure the model to use:

`device = "cpu"`

Use FP32 initially for compatibility and reliability.

Do not assume GPU availability or use automatic GPU device mapping.

## 6. Audio Support

Support these audio formats where practical:

- WAV
- MP3
- M4A
- FLAC

Normalize audio to the sample rate expected by the model, typically 16 kHz.

Handle stereo-to-mono conversion when necessary.

For longer recordings, implement sensible chunking if required by the model's input limitations.

Do not silently truncate long recordings.

## 7. User Interface

For the first version, I only need a simple command-line interface.

Example:

```powershell
python transcribe.py "audio.wav"
```

Expected output:

```text
Loading Shook Tiny...

Processing audio...

Amharic transcription:
[Recognized Amharic text appears here]

Saved to: output.txt
```

Also provide a simple interactive mode where I can enter an audio file path without restarting the application.

Do not build a web application or desktop GUI yet.

## 8. Offline Verification

After downloading the model:

1. Disconnect the application from external model services.
2. Load the model using its local directory.
3. Run a transcription test.
4. Confirm that no API or internet connection is needed for inference.
5. Verify that all model files are stored locally.

The application must not attempt to download anything during normal offline execution.

## 9. Performance Testing

Run a real transcription test using an Amharic audio recording.

Measure and report:

- Model loading time.
- Audio duration.
- Transcription processing time.
- CPU usage if available.
- Peak RAM usage if measurable.
- Whether transcription completes successfully.

Calculate the real-time factor (RTF):

RTF = Processing Time / Audio Duration

An RTF below 1 means transcription is faster than the audio's playback duration.

Do not invent performance numbers. Measure them on this computer.

## 10. Project Structure

Organize the project approximately like this:

```text
Amharic-ASR/
├── models/
│   └── shook-tiny-stage2/
├── audio/
├── output/
├── transcribe.py
├── requirements.txt
├── setup.ps1
├── README.md
└── .gitignore
```

Do not put downloaded model weights into Git.

## 11. Final Verification

Before considering the task complete, verify:

- [ ] Model downloaded successfully.
- [ ] Model loads without CUDA.
- [ ] CPU inference works.
- [ ] Amharic audio is transcribed.
- [ ] Amharic Unicode characters are preserved.
- [ ] Transcription is saved to a text file.
- [ ] Multiple audio formats work where supported.
- [ ] Long recordings are not silently truncated.
- [ ] Offline inference works after model download.
- [ ] Instructions are documented in README.md.

## 12. Important Instructions

Do not overcomplicate this project.

I am not asking you to build an AI platform, train a model, or create a production application.

I simply want to download an existing Amharic speech recognition model, get it running on my CPU, and test its transcription quality.

If something fails, investigate the actual error and fix it rather than replacing the model.

At the end, give me:

1. Confirmation that the model runs.
2. The exact command to start transcription.
3. The location of the downloaded model.
4. A sample transcription result.
5. Actual CPU performance measurements.
6. Any limitations you encountered.

**Priority: Working offline CPU transcription, not unnecessary features.**

## CRITICAL: NO AUTHENTICATION AND FULL OFFLINE OPERATION

The model must be publicly accessible and downloadable without authentication.

- Do not request or require a Hugging Face token.
- Do not require Hugging Face login.
- Do not use gated model repositories.
- Do not require API keys, credentials, or subscriptions.
- Download the model anonymously using publicly accessible files.
- Download all required model weights, tokenizer, processor, and configuration files during setup.
- Store everything locally.

After the initial download, the application must operate completely offline.

Set:

`HF_HUB_OFFLINE=1`

`TRANSFORMERS_OFFLINE=1`

Use local model loading (`local_files_only=True`) during inference.

If the repository requires authentication or cannot be downloaded anonymously, stop and report the issue. Do not request my credentials or silently switch models.

**Final requirement:** I should be able to disconnect my computer from the internet, run the application, provide an Amharic audio file, and receive an Amharic transcription without any external connection.
