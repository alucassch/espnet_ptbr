import os
import io
import copy
import yaml
import torch
import shutil
import zipfile
import librosa
import numpy as np

from pathlib import Path
from os.path import abspath, join, exists, dirname, isfile, isdir
from typing import Tuple, Any, Dict, Union, TypeVar

from espnet2.tasks.asr import ASRTask
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.torch_utils.device_funcs import to_device

from google_drive_downloader import GoogleDriveDownloader as gdd

def download_model(model_type, token_type):
    fname = f"asr_train_commonvoice_{model_type}_raw_{token_type}_valid.acc.ave.zip"
    if not os.path.exists(fname):
        print(f"Downloading {fname}")
        gdd.download_file_from_google_drive(
            file_id=MODELS_URLS[f"{model_type}_{token_type}"],
            dest_path=f"./{fname}"
        )
    else:
        print(f"Model file {fname} exists")
    return fname

MODEL_TYPES = ["rnn", "convrnn", "transformer"]
model_type_d = {"rnn":"rnn", "convrnn":"vggrnn", "transformer":"transformer"}
TOKEN_TYPES = ["char", "bpe"]
token_type_d = {'char': 'char', 'subword': 'bpe'}

MODELS_URLS = {
    "rnn_char" : "1c_J3MEEPQXhaYSYTMy-Pp6Wm7g4F3ppo",
    "vggrnn_char" : "12SAYVc8LMDEg9Hm5vcVnwIw_H1xqH8Dh",
    "transformer_char" : "1Sm_LZkna8RMCxWBCdoZwdPHecxJ5X24F",
    "rnn_bpe" : "18Ges8RBV5VOx1l7EuYUYrl4B7ikln4j-",
    "vggrnn_bpe" : "1A8sLkrP-Gl_BnQV0Nor5o46slmVHQE-t",
    "transformer_bpe" : "1EXhX_mvlZifFdxUXMlI4h-tTyqaJfZov"
}

class Result(object):
    def __init__(self) -> None:
        self.text = None
        self.tokens_txt = None
        self.tokens_int = None
        self.ctc_posteriors = None
        self.attention_weights = None
        self.encoded_vector = None
        self.audio_samples = None
        self.mel_features = None

class ASR(object):
    def __init__(
        self, 
        zip_model_file: Union[Path, str],
    ) -> None:
        
        self.zip_model_file = abspath(zip_model_file)
        self.device = 'cpu'
        self.model = None
        self.beam_search = None
        self.tokenizer = None
        self.converter = None
        self.global_cmvn = None
        self.extract_zip_model_file(self.zip_model_file)
        
    def extract_zip_model_file(self, zip_model_file: str) -> Dict[str, Any]:
      """Extrai os dados de um zip contendo o arquivo com o estado do modelo e configurações

      Args:
          zip_model_file (str): ZipFile do modelo gerado dos scripts de treinamento

      Raises:
          ValueError: Se o arquivo não for correto
          FileNotFoundError: Se o arquivo zip não contiver os arquivos necessários

      Returns:
          Dict[str, Any]: Dicionário do arquivo .yaml utilizado durante o treinamento para carregar o modelo corretamente
      """
      print("Unzipping model")
      if not zipfile.is_zipfile(zip_model_file):
          raise ValueError(f"File {zip_model_file} is not a zipfile")
      else:
          zipfile.ZipFile(zip_model_file).extractall(dirname(zip_model_file))

      check = ['exp', 'meta.yaml']

      if not all([x for x in check]):
          raise FileNotFoundError
      
      print("Load yaml file")
      with open('meta.yaml') as f:
          meta = yaml.load(f, Loader=yaml.FullLoader)

      model_stats_file = meta['files']['asr_model_file']
      asr_model_config_file = meta['yaml_files']['asr_train_config']
      
      self.model_config = {}
      with open(asr_model_config_file) as f:
          self.model_config = yaml.load(f, Loader=yaml.FullLoader)
          try:
              self.global_cmvn = self.model_config['normalize_conf']['stats_file']
          except KeyError:
              self.global_cmvn = None

      print(f'Loading model config from {asr_model_config_file}')
      print(f'Loading model state from {model_stats_file}')

      #Build Model
      print('Building model')
      self.model, _ = ASRTask.build_model_from_file(
          asr_model_config_file, model_stats_file, self.device
      )
      self.model.to(dtype=getattr(torch, 'float32')).eval()
      
      #print("Loading extra modules")
      self.build_beam_search()
      self.build_tokenizer()

    def build_beam_search(self, ctc_weight: float = 0.4, beam_size: int = 1):
        """Constroi o objeto de decodificação beam_search.

        Esse objeto faz a decodificação do vetor de embeddings da saída da parte encoder
        do modelo passando pelos decoders da rede que são o módulo CTC e Transformer ou RNN.

        Como:
        Loss = (1-λ)*DecoderLoss + λ*CTCLoss 
        Se ctc_weight=1 apenas o módulo CTC será usado na decodificação

        Args:
            ctc_weight (float, optional): Peso dado ao módulo CTC da rede. Defaults to 0.4.
            beam_size (int, optional): Tamanho do feixe de busca durante a codificação. Defaults to 1.
        """
        scorers = {}
        ctc = CTCPrefixScorer(ctc=self.model.ctc, eos=self.model.eos)
        token_list = self.model.token_list
        scorers.update(
            decoder=self.model.decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        #Variáveis com os pesos para cada parte da decodificação
        #lm referente à modelos de linguagem não são utilizados aqui mas são necessários no objeto
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=1.0,
            length_bonus=0.0,
        )

        #Cria o objeto beam_search
        self.beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=self.model.sos,
            eos=self.model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )

        self.beam_search.to(device=self.device, dtype=getattr(torch, 'float32')).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=self.device, dtype=getattr(torch, 'float32')).eval()

    def build_tokenizer(self):
        """Cria um objeto tokenizer para conversão dos tokens inteiros para o dicionário
        de caracteres correspondente.

        Caso o modelo possua um modelo BPE de tokenização, ele é utilizado. Se não, apenas a lista
        de caracteres no arquivo de configuração é usada.
        """
        token_type = self.model_config['token_type']
        if token_type == 'bpe':
            bpemodel = self.model_config['bpemodel']
            self.tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            self.tokenizer = build_tokenizer(token_type=token_type)
        
        self.converter = TokenIDConverter(token_list=self.model.token_list)

    def get_layers(self) -> Dict[str, Dict[str, torch.Size]]:
        """Retorna as camadas nomeadas e os respectivos shapes para todos os módulos da rede.

        Os módulos são:
            Encoder: RNN, VGGRNN, TransformerEncoder
            Decoder: RNN, TransformerDecoder
            CTC

        Returns:
            Dict[str, Dict[str, torch.Size]]: Dicionário de cada módulo com seus respectivos layers e shape
        """
        r = {}

        r['frontend'] = {x: self.model.frontend.state_dict()[x].shape 
                            for x in self.model.frontend.state_dict().keys()}
        r['specaug'] = {x: self.model.specaug.state_dict()[x].shape 
                            for x in self.model.specaug.state_dict().keys()}
        r['normalize'] = {x: self.model.normalize.state_dict()[x].shape 
                            for x in self.model.normalize.state_dict().keys()}
        r['encoder'] = {x: self.model.encoder.state_dict()[x].shape 
                            for x in self.model.encoder.state_dict().keys()}
        r['decoder'] = {x: self.model.decoder.state_dict()[x].shape 
                            for x in self.model.decoder.state_dict().keys()}
        r['ctc']     = {x: self.model.ctc.state_dict()[x].shape 
                            for x in self.model.ctc.state_dict().keys()}
        return r

    def frontend(self, audiofile: Union[Path, str, bytes], normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa o frontend do modelo, transformando as amostras de áudio em parâmetros log mel spectrogram

        Args:
            audiofile (Union[Path, str]): arquivo de áudio

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Parâmetros, Tamanho do vetor de parâmetros
        """
        if isinstance(audiofile, str):
            audio_samples, rate = librosa.load(audiofile, sr=16000)
        elif isinstance(audiofile, bytes):
            audio_samples, rate = librosa.core.load(io.BytesIO(audiofile), sr=16000)
        else:
            raise ValueError("Failed to load audio file")
            
        if isinstance(audio_samples, np.ndarray):
            audio_samples = torch.tensor(audio_samples)
        audio_samples = audio_samples.unsqueeze(0).to(getattr(torch, 'float32'))
        lengths = audio_samples.new_full([1], dtype=torch.long, fill_value=audio_samples.size(1))
        features, features_length  = self.model.frontend(audio_samples, lengths)

        if normalize:
            features, features_length = self.model.normalize(features, features_length)

        return features, features_length

    def specaug(self, features: torch.Tensor, features_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa o módulo specaug, da parte de 'data augmentation'.
        Útil para visualização apenas. 
        Não é utilizado na inferência, apenas no treinamento.

        Args:
            features (torch.Tensor): Parâmetros
            features_length (torch.Tensor): tamanho do vetor de parâmetros

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Parâmetros com máscaras temporais, em frequência e distoção. Tamanho dos vetores
        """
        return self.model.specaug(features, features_length)

    def __del__(self) -> None:
        """Remove os arquivos temporários
        """
        for f in ['exp', 'meta.yaml']:
            print(f"Removing {f}")
            ff = join(dirname(self.zip_model_file), f)
            if exists(ff):
                if isdir(ff):
                    shutil.rmtree(ff)
                elif isfile(ff):
                    os.remove(ff)
                else:
                    raise ValueError("Error ao remover arquivos temporários")
            

    @torch.no_grad()
    def recognize(self, audiofile: Union[Path, str, bytes]) -> Result:

        result = Result()
        
        if isinstance(audiofile, str):
            audio_samples, rate = librosa.load(audiofile, sr=16000)
        elif isinstance(audiofile, bytes):
            audio_samples, rate = librosa.core.load(io.BytesIO(audiofile), sr=16000)
        else:
            raise ValueError("Failed to load audio file")

        result.audio_samples = copy.deepcopy(audio_samples)

        #a entrada do modelo é torch.tensor
        if isinstance(audio_samples, np.ndarray):
            audio_samples = torch.tensor(audio_samples)
        audio_samples = audio_samples.unsqueeze(0).to(getattr(torch, 'float32'))
        
        lengths = audio_samples.new_full([1], dtype=torch.long, fill_value=audio_samples.size(1))
        batch = {"speech": audio_samples, "speech_lengths": lengths}
        batch = to_device(batch, device=self.device)

        #model encoder
        enc, _ = self.model.encode(**batch)

        #model decoder
        nbest_hyps = self.beam_search(x=enc[0])

        #Apenas a melhor hipótese
        best_hyps = nbest_hyps[0]

        #Conversão de tokenids do treinamento para texto
        token_int = best_hyps.yseq[1:-1].tolist()
        token_int = list(filter(lambda x: x != 0, token_int))
        token = self.converter.ids2tokens(token_int)
        text = self.tokenizer.tokens2text(token)

        #Preenche o objeto result
        result.text = text
        result.encoded_vector = enc[0] #[0] remove dimensão de batch
        
        #calcula todas as matrizes de atenção
        #
        text_tensor = torch.Tensor(token_int).unsqueeze(0).to(getattr(torch, 'long'))
        batch["text"] = text_tensor
        batch["text_lengths"] = text_tensor.new_full([1], dtype=torch.long, fill_value=text_tensor.size(1))
        
        result.attention_weights = calculate_all_attentions(self.model, batch)
        result.tokens_txt = token

        #CTC posteriors
        logp = self.model.ctc.log_softmax(enc.unsqueeze(0))[0]
        result.ctc_posteriors = logp.exp_().numpy()
        result.tokens_int = best_hyps.yseq
        result.mel_features, _ = self.frontend(audiofile, normalize=False)
        return result

    def __call__(self, input: Union[Path, str, bytes]) -> Result:
        return self.recognize(input)


if __name__ == '__main__':

    #model_type: rnn, convrnn, transformer
    model_type = 'transformer'
    #token_type = char, subword
    token_type = 'char'

    model_type = model_type_d[model_type]
    token_type = token_type_d[token_type]
    asr_tag = "asr_train_commonvoice_"+model_type+"_raw_"+token_type

    #Faz o downlooad do modelo caso o arquivo ainda não exista
    model_file = download_model(model_type, token_type)

    #A classe ASR encapsula um modelo ESPNet em asr.model
    asr = ASR(model_file)

    #O método recognize recebe um arquivo de áudio ou um buffer de memória e fornece a melhor hipotese do modelo
    audio_file = join(dirname(__file__), 'teste_pesquisa.wav')
    results = asr.recognize(audio_file)

    print(f"\n\nHipotese: {results.text}\n\n")