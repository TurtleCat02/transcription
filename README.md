# How to use:

Create a python environment (this program was written in Python 3.12, earlier 3.xx versions may work)

If you want to use with a GPU (which you should, only supports NVIDIA GPUs) run:
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

Run `pip install -r requirements.txt`

Install ffmpeg by running `winget install "FFmpeg (Essentials Build)"`
or downloading [this](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z) and adding the bin folder to your windows path

Add your audio file to `.\audio\` and run the diarizer and then the transcriber
`python .\diarizer.py .\audio\<AUDIO_FILE> -n <NUM_SPEAKERS>`
`python .\transcriber.py .\audio\<AUDIO_FILE> -m <MODEL>`
(Replace values inside <>, and do not include brackets)

You don't need to specify a number of speakers, and it will guess, but it might not be good.
If using a GPU, use the `medium` or `large` models, if on CPU use `base` or `small` (Larger the model, better the output will be)

transcriber.py automatically outputs transcripts to `.\transcripts\<AUDIO_FILE>.txt`


Add -h (`python .\diarizer -h`, `python .\transcriber -h`) for more options