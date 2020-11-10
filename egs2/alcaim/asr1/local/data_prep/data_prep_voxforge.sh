#!/bin/bash

# Copyright 2020  André Schlichting
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/b9/strange_star/sid data/sid"
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

python3 local/data_prep/prep_voxforge.py $src $dst || exit 1

utils/fix_data_dir.sh $dst || exit 1
utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
