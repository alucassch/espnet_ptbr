import os
import sys
from glob import glob

from typing import List

def process_text(text: str) -> str:
    return text.lower().replace(',','').replace('.','').replace('-','')

def main(args: List[str]):
    src, dst = None, None
    if len(args) != 3:
        print(f"Usage {__file__} src_dir dst_dir")
        sys.exit(1)
    else:
        src, dst = args[1:]
        if not os.path.exists(dst): os.makedirs(dst)

    try:
        fwav  = open(os.path.join(dst, 'wav.scp'), 'w')
        ftext = open(os.path.join(dst, 'text'), 'w')
        futt2spk = open(os.path.join(dst, 'utt2spk'), 'w')
        fskp2utt = open(os.path.join(dst, 'spk2utt'), 'w')

        for spkr_name in os.listdir(src):
            prompts = {}
            with open(os.path.join(src, spkr_name, 'prompts.txt')) as f:
                for line in f:
                    uid, text = line.strip().split('=')
                    prompts[int(uid)] = process_text(text)
            for audio in glob(os.path.join(src, spkr_name, '*.wav')):
                basename = os.path.splitext(os.path.basename(audio))[0]
                uttid = int(basename[-3:])
                if uttid in prompts.keys():
                    sox_cmd = f"sox -t wav {os.path.abspath(audio)} -t wav -r 16000 - |"
                    fwav.write(f"{spkr_name}-{basename} {sox_cmd}\n")
                    ftext.write(f"{spkr_name}-{basename} {prompts[uttid]}\n")
                    futt2spk.write(f"{spkr_name}-{basename} {spkr_name}\n")
                    fskp2utt.write(f"{spkr_name} {spkr_name}-{basename}\n")
    finally:
        fwav.close()
        ftext.close()
        futt2spk.close()
        fskp2utt.close()


if __name__ == '__main__': main(sys.argv)