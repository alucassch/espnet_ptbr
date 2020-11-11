import os
import sys
from glob import glob

from typing import List

def process_text(text: str) -> str:
    #"" is some strange unicode char
    return text.lower().replace("","").replace("-"," ")

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
            spkid = spkr_name.split('_')[-1]
            for audio in glob(os.path.join(src, spkr_name, '*.wav')):
                basename = os.path.splitext(os.path.basename(audio))[0]
                with open(os.path.join(src, spkr_name, basename+'.txt')) as f:
                    text = process_text(f.readlines()[0].strip())

                sox_cmd = f"sox -t wav {os.path.abspath(audio)} -t wav -r 16000 - |"
                fwav.write(f"{spkid}-{basename} {sox_cmd}\n")
                ftext.write(f"{spkid}-{basename} {text}\n")
                futt2spk.write(f"{spkid}-{basename} {spkid}\n")
                fskp2utt.write(f"{spkid} {spkid}-{basename}\n")
    finally:
        fwav.close()
        ftext.close()
        futt2spk.close()
        fskp2utt.close()


if __name__ == '__main__': main(sys.argv)