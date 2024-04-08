from argparse import ArgumentParser
from pathlib import Path

from pydub import AudioSegment

FILENAME = "Febe_Distributor_Taichung_Interview_26_03"
EMBED = "ecapa"
CLUSTER = "sc"

EXTRA = ""  # f"_{CLUSTER}"
SEGMENT_FILE = f"./segments/{FILENAME}{EXTRA}.rttm"

TONE_FILE = f"./audio/440Hz_44100Hz_16bit_05sec.wav"
TONE = AudioSegment.from_wav(TONE_FILE)[:500]
SILENCE = AudioSegment.silent(duration=500)

BREAK = SILENCE

speaker_audio = {}
speaker_prev = {}
with open(SEGMENT_FILE, "r") as f:
    audio = AudioSegment.from_wav(f"./audio/{FILENAME}.wav")
    for line in f:

        data = line.strip().split(sep=" ")

        speaker = data[7]
        start = float(data[3]) * 1000
        end = start + (float(data[4]) * 1000)
        if speaker_audio.get(speaker):
            speaker_audio[speaker] = speaker_audio[speaker] + AudioSegment.silent(start - speaker_prev[speaker]) + audio[start:end]
            speaker_prev[speaker] = end
        else:
            speaker_audio[speaker] = AudioSegment.silent(start) + audio[start:end]
            speaker_prev[speaker] = end

Path(f"./diarized/{FILENAME}{EXTRA}/").mkdir(parents=True)
for k, v in speaker_audio.items():
    v.export(f"./diarized/{FILENAME}{EXTRA}/{k}.m4a", format="m4a")

# with open(SEGMENT_FILE, "r") as f:
#     audio = AudioSegment.from_wav(f"./audio/{FILENAME}.wav")
#     for line in f:
#         data = line.strip().split(sep=" ")
#         speaker = data[7]
#         start = float(data[3]) * 1000
#         end = start + (float(data[4]) * 1000)
#         if speaker_audio.get(speaker):
#             speaker_audio[speaker] = speaker_audio[speaker] + BREAK + audio[start:end]
#         else:
#             speaker_audio[speaker] = audio[start:end]
#
# os.makedirs(f"./diarized/{FILENAME}{EXTRA}/")
# for k, v in speaker_audio.items():
#     v.export(f"./diarized/{FILENAME}{EXTRA}/{k}.wav", format="wav")
