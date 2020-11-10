#!/bin/bash
# Set bash to 'debug' mode,it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
data_url=http://www02.smt.ufrj.br/~igor.quintanilha
data_url_cv=https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22
train_set=train
train_dev=dev
recog_set="lapsbm-val lapsbm-test voxforge-ptbr"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${ALCAIM}" ]; then
    log "Fill the value of 'ALCAIM' of db.sh"
    exit 1
fi

if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "stage 1: Data Download to ${ALCAIM}"
	for part in lapsbm-val lapsbm-test voxforge-ptbr alcaim sid; do
            local/download_and_untar.sh ${ALCAIM} ${data_url} ${part}
	done
    echo "stage 1: Data Download to ${COMMONVOICE}"
    local/download_and_untar.sh ${COMMONVOICE} ${data_url_cv} pt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    local/data_prep/data_prep_lapsbm.sh ${ALCAIM}/lapsbm-val data/lapsbm-val
    local/data_prep/data_prep_lapsbm.sh ${ALCAIM}/lapsbm-test data/lapsbm-test
    local/data_prep/data_prep_sid.sh ${ALCAIM}/sid data/sid
    local/data_prep/data_prep_alcaim.sh ${ALCAIM}/alcaim/alcaim data/alcaim
    local/data_prep/data_prep_voxforge.sh ${ALCAIM}/voxforge-ptbr data/voxforge-ptbr
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage3: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep/data_prep_voxforge.pl "${COMMONVOICE}/pt/cv-corpus-5.1-2020-06-22/${lang}" ${part} data/commonvoice_"$(echo "${part}_${lang}" | tr - _)"
    done

    train_set=commonvoice_train_"$(echo "${lang}" | tr - _)"
    train_dev=commonvoice_dev_"$(echo "${lang}" | tr - _)"
    test_set=commonvoice_test_"$(echo "${lang}" | tr - _)"

    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh data/commonvoice_"$(echo "validated_${lang}" | tr - _)" data/${train_set}
    utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp
    utils/fix_data_dir.sh data/${train_set}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/alcaim
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/sid
    utils/combine_data.sh --extra_files utt2num_frames data/lapsbm data/lapsbm-val data/lapsbm-test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
