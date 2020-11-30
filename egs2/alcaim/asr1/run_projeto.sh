#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

datasets="commonvoice"
model_types="rnn vggrnn transformer"
feature_types="raw"
token_types="char bpe"

inference_config=conf/decode_asr.yaml

stage=1

for dataset in ${datasets}; do
    for feature_type in ${feature_types}; do
        for model_type in ${model_types}; do
    
            train_set="commonvoice_train_pt"
            valid_set="commonvoice_dev_pt"
            test_sets="commonvoice_test_pt lapsbm voxforge sid"

            if [ "${model_type}" = vggrnn ]; then
                asr_config=conf/train_testes/train_asr_vggrnn_${dataset}.yaml
            elif [ "${model_type}" = rnn ]; then
                asr_config=conf/train_testes/train_asr_rnn_${dataset}.yaml
            elif [ "${model_type}" = transformer ]; then
                asr_config=conf/train_testes/train_asr_transformer_${dataset}.yaml
            fi


            for token_type in ${token_types}; do
                asr_tag="train_${dataset}_${model_type}_${feature_type}_${token_type}"
                echo "$asr_tag" > wip.txt
                if [ ! -f "${asr_tag}.done" ]; then   
                    ./asr.sh \
                        --stage "${stage}" \
                        --asr_tag "${asr_tag}" \
                        --nj 400 \
                        --inference_nj 300 \
                        --lang pt \
                        --ngpu 2 \
                        --feats_type ${feature_type} \
                        --audio_format wav \
                        --token_type ${token_type} \
                        --nbpe 500 \
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
    done
done
