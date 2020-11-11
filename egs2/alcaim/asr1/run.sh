#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="lapsbm voxforge-ptbr commonvoice-ptbr"

asr_config=conf/tuning/train_asr_transformer.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --nj 12 \
    --lang pt \
    --ngpu 1 \
    --feats_type fbank_pitch \
    --token_type bpe \
    --bpemode bpe \
    --nbpe 1000 \
    --use_lm false \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" "$@"
