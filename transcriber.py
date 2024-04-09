import os
from argparse import ArgumentParser
from pathlib import Path

import whisper

MODEL = "base"
SAMPLE_RATE = 16000

WHISPER_SPEECH_THRESH = 0.6
WHISPER_AVGLOGPROB_THRESH = -1

SOFT_AVGLOGPROB_THRESH = -1.75

HARD_SPEECH_THRESH = 0.8
HARD_AVGLOGPROB_THRESH = -2.5
UNINTELLIGIBLE_SPEECH_THRESH = 0.5


def transcribe(audio_file, segment_file, output):
    if segment_file is None:
        segment_file = f"./segments/{Path(audio_file).stem}.rttm"
    if output is None:
        output = f"./transcripts/{Path(audio_file).stem}.txt"

    logfile = f"./logs/{Path(output).stem}.txt"
    model = whisper.load_model(MODEL)
    audio = whisper.load_audio(audio_file)

    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    prev_speaker = None
    prev_lang = None
    seg_start = 0
    seg_end = 0
    with open(segment_file, "r") as segments, open(output, "wb") as out, open(logfile, "wb") as log:
        for line in segments:
            data = line.strip().split(sep=" ")

            speaker = data[7]
            start = int(float(data[3]) * SAMPLE_RATE)
            end = start + int(float(data[4]) * SAMPLE_RATE)

            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio[start:end])).to(model.device)
            _, probs = model.detect_language(mel)
            lang = "en" if probs["en"] > probs["zh"] else "zh"

            if (speaker == prev_speaker) and (lang == prev_lang):
                seg_end = end
                continue
            elif not (prev_speaker and prev_lang):
                prev_speaker = speaker
                prev_lang = lang
                seg_end = end
                continue

            transcript = model.transcribe(audio[seg_start:seg_end], no_speech_threshold=WHISPER_SPEECH_THRESH,
                                          logprob_threshold=WHISPER_AVGLOGPROB_THRESH, language=prev_lang)
            write_transcript(out, log, prev_speaker, transcript, seg_start, seg_end)
            prev_lang = lang
            prev_speaker = speaker
            seg_start = start
            seg_end = end

        transcript = model.transcribe(audio[seg_start:seg_end], no_speech_threshold=WHISPER_SPEECH_THRESH,
                                      logprob_threshold=WHISPER_AVGLOGPROB_THRESH, language=prev_lang)
        write_transcript(out, log, prev_speaker, transcript, seg_start, seg_end)


def write_transcript(out, log, speaker, transcript, seg_start, seg_end):
    if not transcript['segments']:
        return
    start_m, start_s = divmod(float(seg_start / SAMPLE_RATE), 60)
    end_m, end_s = divmod(float(seg_end / SAMPLE_RATE), 60)
    out.write(f"[{int(start_m):0>2d}:{int(start_s):0>2d}->{int(end_m):0>2d}:{int(end_s):0>2d}] Speaker {speaker}: ".encode("utf-8"))
    for segment in transcript["segments"]:
        start_m, start_s = divmod(float((seg_start / SAMPLE_RATE) + segment['start']), 60)
        end_m, end_s = divmod(float((seg_start / SAMPLE_RATE) + segment['end']), 60)
        text = segment['text'].strip()
        log.write(f"({segment['no_speech_prob']:.3f},{segment['avg_logprob']:.3f}) [{int(start_m):0>2d}:{int(start_s):0>2d}->{int(end_m):0>2d}:{int(end_s):0>2d}] Speaker {speaker}: ".encode("utf-8"))
        print(f"[{int(start_m):0>2d}:{int(start_s):0>2d}->{int(end_m):0>2d}:{int(end_s):0>2d}] Speaker {speaker}: ", end="")
        if segment["no_speech_prob"] > HARD_SPEECH_THRESH:
            log.write(f"(SUPPRESSED-NO SPEECH) ".encode("utf-8"))
            print(f"(SUPPRESSED-NO SPEECH)")
        elif segment["avg_logprob"] < SOFT_AVGLOGPROB_THRESH:
            if segment["avg_logprob"] < HARD_AVGLOGPROB_THRESH:
                if segment["no_speech_prob"] < UNINTELLIGIBLE_SPEECH_THRESH:
                    out.write("|UNINTELLIGIBLE| ".encode("utf-8"))
                    log.write("|UNINTELLIGIBLE| ".encode("utf-8"))
                    print(f"|UNINTELLIGIBLE|")
                else:
                    log.write("(SUPPRESSED-UNINTELLIGIBLE) ".encode("utf-8"))
                    print(f"(SUPPRESSED-UNINTELLIGIBLE)")

            else:
                out.write(f"(LOW CONFIDENCE) {text} ".encode("utf-8"))
                log.write("(LOW CONFIDENCE) ".encode("utf-8"))
                print(f"(LOW CONFIDENCE) {text}")

        else:
            out.write(f"{text} ".encode("utf-8"))
            print(f"{text}")

        log.write(f"{text}\n".encode("utf-8"))

    out.write("\n".encode("utf-8"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Transcription using whisper")
    parser.add_argument("audio_file")
    parser.add_argument('-s', '--segment_file')
    parser.add_argument('-m', '--model', choices=["base", "small", "medium", "large"], default="base")
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    transcribe(args.audio_file, args.segment_file, args.output)
