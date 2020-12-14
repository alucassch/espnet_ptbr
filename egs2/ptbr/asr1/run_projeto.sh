#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="commonvoice_train_pt"
valid_set="commonvoice_dev_pt"
test_sets="commonvoice_test_pt lapsbm voxforge sid"

model_types="rnn vggrnn transformer"
token_types="char bpe"
nbpe=500

inference_config=conf/decode_asr.yaml

for model_type in ${model_types}; do

    if [ "${model_type}" = rnn ]; then
        asr_config=conf/train_asr_rnn_commonvoice.yaml
    elif [ "${model_type}" = vggrnn ]; then
        asr_config=conf/train_asr_vggrnn_commonvoice.yaml
    elif [ "${model_type}" = transformer ]; then
        asr_config=conf/train_asr_transformer_commonvoice.yaml
    fi

    for token_type in ${token_types}; do
        asr_tag="train_commonvoice_${model_type}_raw_${token_type}"
        echo "$asr_tag" > wip.txt
        if [ ! -f "${asr_tag}.done" ]; then   
            ./asr.sh \
                --stage 1 \
                --asr_tag "${asr_tag}" \
                --nj 400 \
                --inference_nj 300 \
                --lang pt \
                --ngpu 2 \
                --feats_type raw \
                --audio_format wav \
                --token_type "${token_type}" \
                --nbpe "${nbpe}" \
                --bpemode bpe \
                --use_lm false \
                --max_wav_duration 30 \
                --asr_config "${asr_config}" \
                --inference_config "${inference_config}" \
                --train_set "${train_set}" \
                --valid_set "${valid_set}" \
                --test_sets "${test_sets}" \
                --srctexts "data/${train_set}/text" "$@"
            touch "${asr_tag}".done
        else
            echo "skip ${asr_tag}"
        fi
    done
done
