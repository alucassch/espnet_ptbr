import os
import sys
from glob import glob
from os.path import join, exists

from typing import List, Dict

FILTER_OUT = ['anonymous-20140619-wcy',
            'ThiagoCastro-20131129-qpn',
            'Marcelo-20131106-iqc',
            'anonymous-20131016-uzv']

def process_text(text: str) -> str:
    return text.lower().replace('.','').replace(',','').replace('?','').replace('-',' ').replace('!','')

def get_prompts(d: str) -> Dict[int, str]:
    prompts = {}
    with open(join(d,'prompts-original')) as f:
        for line in f:
            utt = int(line.split()[0])
            text = process_text(' '.join(line.strip().split()[1:])).strip()
            prompts[utt] = text

    return prompts

def get_wavs(d: str) -> Dict[int, str]:
    wavs = {}
    for f in glob(join(d, '*.wav')):
        uttid = int(os.path.splitext(os.path.basename(f))[0])
        wavs[uttid] = os.path.abspath(f)

    return wavs

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

        for i, d in enumerate(os.listdir(src)):
            if not d in FILTER_OUT:
                spkid = str(i)+'-'+ os.path.basename(join(src,d)).split('-')[0]
                if exists(join(src, d, 'etc')) and exists(join(src, d, 'wav')):
                    prompts = get_prompts(join(src, d, 'etc'))
                    wavs = get_wavs(join(src, d, 'wav'))
                else:
                    prompts = get_prompts(join(src, d))
                    wavs = get_wavs(join(src, d))

                match_keys = set(prompts.keys()) & set(wavs.keys())

                for k in match_keys:
                    wav_file = wavs[k]
                    sox_cmd = f"sox -t wav {wav_file} -t wav -r 16000 - |"
                    basename = os.path.splitext(os.path.basename(wav_file))[0]
                    fwav.write(f"{spkid}-{basename} {sox_cmd}\n")
                    ftext.write(f"{spkid}-{basename} {prompts[k]}\n")
                    futt2spk.write(f"{spkid}-{basename} {spkid}\n")
                    fskp2utt.write(f"{spkid} {spkid}-{basename}\n")
    finally:
        fwav.close()
        ftext.close()
        futt2spk.close()
        fskp2utt.close()


if __name__ == '__main__': main(sys.argv)