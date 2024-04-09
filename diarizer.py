import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

EMBED = "ecapa"
CLUSTER = "sc"
WINDOW = 1.5
PERIOD = 0.75


def diarize(audio_file, num_speakers=None, outfile=None, embed=EMBED, cluster=CLUSTER, window=WINDOW, period=PERIOD):
    import soundfile as sf
    from simple_diarizer.diarizer import Diarizer
    from simple_diarizer.utils import (convert_wavfile, combined_waveplot)
    filepath = Path(audio_file)
    converted = filepath.with_name(f"{filepath.stem}_converted.wav")
    if not converted.exists():
        convert_wavfile(filepath, converted)
    signal, fs = sf.read(converted)
    diar = Diarizer(
        embed_model=embed,  # supported types: ['xvec', 'ecapa']
        cluster_method=cluster,  # supported types: ['ahc', 'sc']
        window=window,  # size of window to extract embeddings (in seconds)
        period=period  # hop of window (in seconds)
    )
    if outfile is None:
        outfile = f".\\segments\\{filepath.stem}.rttm"
    else:
        outfile = Path(outfile).with_suffix(".rttm")

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    segments = diar.diarize(converted,
                            num_speakers=num_speakers,
                            outfile=outfile,
                            silence_tolerance=0.1)
    os.makedirs("./plots/", exist_ok=True)
    combined_waveplot(signal, fs, segments, figsize=(10, 3), tick_interval=60)
    plt.savefig(f".\\plots\\{Path(outfile).stem}.png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Speech diarization")
    parser.add_argument("audio_file")
    parser.add_argument('-n', "--num_speakers", type=int)
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-e', '--embed', default=EMBED, choices=["xvec", "ecapa"])
    parser.add_argument('-c', '--cluster', default=CLUSTER, choices=["ahc", "sc"])
    parser.add_argument('-w', '--window', default=WINDOW, type=float)
    parser.add_argument('-p', '--period', default=PERIOD, type=float)
    args = parser.parse_args()
    diarize(args.audio_file, args.num_speakers, args.outfile, args.embed, args.cluster, args.window, args.period)
