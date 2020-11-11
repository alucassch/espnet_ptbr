#!/bin/bash

# Copyright 2020  André Schlichting
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/b9/strange_star/lapsbm-test data/lapsbm-test"
  exit 1
fi

src=$1
dst=$2

echo "Working on $src"

# use sox to do some preprocessing
if ! which sox >&/dev/null; then
   echo "Please install 'sox'"
   exit 1
fi

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2utt=$dst/spk2utt; [[ -f "$spk2utt" ]] && rm $spk2utt

for spkr_dir in $(find $src -mindepth 1 -maxdepth 1 -type d); do
    spkr_name=$(basename $spkr_dir | awk -F - '{print $2}')
    for utt in $(find $spkr_dir -iname "*.wav"); do
        filename=$(basename -- "$utt")
        filename="${filename%.*}"
        wav_file=$(realpath $spkr_dir/$filename.wav)
        transcription=$(cat `realpath $spkr_dir/$filename.txt` | awk '{print tolower($0)}') 
        echo "$spkr_name-$filename sox -t wav $wav_file -t wav -r 16000 - | " >> $wav_scp
        echo "$spkr_name-$filename $transcription" >> $trans
        echo "$spkr_name-$filename $spkr_name" >> $utt2spk
        echo "$spkr_name $spkr_name-$filename" >> $spk2utt
    done
done

utils/fix_data_dir.sh $dst || exit 1
utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
