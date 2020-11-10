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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${ALCAIM}/download.done" ]; then
	echo "stage 1: Data Download to ${ALCAIM}"
	for part in lapsbm-val lapsbm-test voxforge-ptbr alcaim sid; do
            local/download_and_untar.sh ${ALCAIM} ${data_url} ${part}
	done
    touch ${ALCAIM}/download.done
    else
        log "stage 1: ${ALCAIM}/download.done is already existing. Skip data downloading"
    fi
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
    log "stage 3: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/alcaim
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/sid
    utils/combine_data.sh --extra_files utt2num_frames data/lapsbm data/lapsbm-val data/lapsbm-test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
