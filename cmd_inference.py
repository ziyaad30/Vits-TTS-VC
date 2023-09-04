from pathlib import Path
import utils
from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
from text import text_to_sequence, _clean_text
import commons
from scipy.io.wavfile import write as write_wav
import os
from text.symbols import symbols
import nltk
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, symbols, [] if is_symbol else hps.data.text_cleaners)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def save_wav(audio_array, filename, sample_rate):
    write_wav(filename, sample_rate, audio_array)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='vits inference')

    parser.add_argument('-m', '--model_path', type=str, default="OUTPUT_MODEL/G_latest.pth")
    parser.add_argument('-c', '--config_path', type=str, default="OUTPUT_MODEL/config.json")
    parser.add_argument('-o', '--output_path', type=str, default="output")
    parser.add_argument('-s', '--spk', type=str, default='ziyaad')
    parser.add_argument('-on', '--output_name', type=str, default="audio")
    parser.add_argument('-ns', '--noise_scale', type=float, default= .667)
    parser.add_argument('-nsw', '--noise_scale_w', type=float, default=0.6)
    parser.add_argument('-ls', '--length_scale', type=float, default=0.8)
    
    args = parser.parse_args()
    
    model_path = args.model_path
    config_path = args.config_path
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    text = """
    His most significant murder came when the commission ordered the hit on dutch schultz. 
    This whole time, louie was hiding, in new york city.
    """
    
    spk = args.spk
    noise_scale = args.noise_scale
    noise_scale_w = args.noise_scale_w
    length = args.length_scale
    output_name = args.output_name
    
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    
    speaker_ids = hps.speakers

    speaker_id = speaker_ids[spk]
    
    sent_text = nltk.sent_tokenize(text)
    
    test_parts = []
    silenceshort = np.zeros(int((float(250) / 1000.0) * 24000), dtype=np.int16)
    
    for j, text in enumerate(sent_text):
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                length_scale=1.0 / length)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid

        audio_array = np.array(audio)
        test_parts += [audio_array]
        test_parts += [silenceshort.copy()]
        
    save_wav(np.concatenate(test_parts), f'{output_dir}/audio.wav', hps.data.sampling_rate)
