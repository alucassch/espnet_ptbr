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

if [ -z "${VOXFORGE}" ]; then
    log "Fill the value of 'VOXFORGE' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "stage 1: Data Download to ${ALCAIM}"
    for part in lapsbm-val lapsbm-test voxforge-ptbr alcaim sid; do
            local/download_and_untar.sh ${ALCAIM} ${data_url} ${part}
	done
    echo "stage 1: Data Download to ${COMMONVOICE}"
    local/download_and_untar.sh ${COMMONVOICE} ${data_url_cv} pt

    if [ ! -e "${VOXFORGE}/voxforge/pt/extracted" ]; then
        echo "stage 1: Data Download to ${VOXFORGE}"
        local/getdata.sh pt ${VOXFORGE}/voxforge
    else
        log "stage 1: ${VOXFORGE}/voxforge/pt/extracted is already existing. Skip data downloading"
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Preparing data for Alcaim, LapsBM & Sid"
    local/data_prep_lapsbm.sh ${ALCAIM}/lapsbm-val data/lapsbm-val
    local/data_prep_lapsbm.sh ${ALCAIM}/lapsbm-test data/lapsbm-test
    local/data_prep_sid.sh ${ALCAIM}/sid data/sid
    local/data_prep_alcaim.sh ${ALCAIM}/alcaim/alcaim data/alcaim
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep_commonvoice.pl "${COMMONVOICE}/pt/cv-corpus-5.1-2020-06-22/pt" ${part} data/commonvoice_"$(echo "${part}_pt" | tr - _)"
    done

    train_set_cv=commonvoice_train_pt
    train_dev_cv=commonvoice_dev_pt
    test_set_cv=commonvoice_test_pt

    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh data/commonvoice_validated_pt data/${train_set_cv}
    utils/filter_scp.pl --exclude data/${train_dev_cv}/wav.scp data/${train_set_cv}/wav.scp > data/${train_set_cv}/temp_wav.scp
    utils/filter_scp.pl --exclude data/${test_set_cv}/wav.scp data/${train_set_cv}/temp_wav.scp > data/${train_set_cv}/wav.scp
    
    python local/postproc_text.py data/commonvoice_validated_pt/text
    python local/postproc_text.py data/${train_set_cv}/text
    python local/postproc_text.py data/${train_dev_cv}/text
    python local/postproc_text.py data/${train_dev_cv}/text
    
    utils/fix_data_dir.sh data/${train_set_cv}

    utils/combine_data.sh --extra_files utt2num_frames data/commonvoice-ptbr data/${train_set_cv} data/${train_dev_cv} data/${train_dev_cv}

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Preparing data for voxforge"
    selected=${VOXFORGE}/voxforge/pt/extracted
    # Initial normalization of the data
    local/voxforge_data_prep.sh --flac2wav true "${selected}" "pt"
    local/voxforge_format_data.sh "pt"

    log "stage 4: Split all_pt into data/voxforge_tr_pt data/voxforge_dt_pt data/voxforge_et_pt"
    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_pt data/voxforge_tr_pt data/voxforge_dt_pt data/voxforge_et_pt
    rm -rf data/all_pt data/local

    python local/postproc_text.py data/voxforge_tr_pt/text
    python local/postproc_text.py data/voxforge_dt_pt/text
    python local/postproc_text.py data/voxforge_et_pt/text

    utils/fix_data_dir.sh data/voxforge_tr_pt
    utils/fix_data_dir.sh data/voxforge_dt_pt
    utils/fix_data_dir.sh data/voxforge_et_pt

    utils/combine_data.sh --extra_files utt2num_frames data/voxforge-ptbr data/voxforge_tr_pt data/voxforge_dt_pt data/voxforge_et_pt

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/alcaim
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/sid
    utils/combine_data.sh --extra_files utt2num_frames data/lapsbm data/lapsbm-val data/lapsbm-test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
