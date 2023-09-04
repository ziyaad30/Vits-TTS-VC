import os
import argparse
import json
import sys
sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.

# load phonemizer
from phonemizer.backend import EspeakBackend

if os.name == 'nt':
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    _ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'    # For Windows
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

def phoneme_text(text):
    backend = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=False, punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')
    text = backend.phonemize([text], strip=True)[0]
    return text.strip()

def fix_text(text):
    text = text.replace('cattroni', 'cotroni')
    text = text.replace('catroni', 'cotroni')
    text = text.replace('catrone', 'cotroni')
    text = text.replace('galanti', 'galante')
    text = text.replace('dalante', 'galante')
    text = text.replace('dalatti', 'galante')
    text = text.replace('vellante', 'galante')
    text = text.replace('magadino', 'magaddino')
    text = text.replace('maliaco', 'magliocco')
    text = text.replace('j. edgar hoover', 'j edgar hoover')
    text = text.replace('buckelter', 'buchalter')
    text = text.replace('lepke-bucklter', 'lepke buchalter')
    text = text.replace('audiorgen', 'augie orgen')
    text = text.replace('augie origin', 'augie orgen')
    text = text.replace('inc.', 'incorporated')
    text = text.replace('levees', 'levies')
    text = text.replace('legge', 'leg')
    text = text.replace('murder, incorporated', 'murder incorporated')
    text = text.replace('penned', 'pinned')
    text = text.replace('delaunay', 'galante')
    text = text.replace('amado', 'amato')
    text = text.replace('bonventura', 'bonventre')
    text = text.replace('lipky', 'lepke')
    text = text.replace('lip key', 'lepke')
    text = text.replace('morgan', 'orgen')
    text = text.replace('midminus one thousand, nine hundred twenty-three', 'mid nineteen twenty-three')
    return text

if __name__ == "__main__":
    new_annos = []
    # Source 1: transcribed short audios
    if os.path.exists("short_character_anno.txt"):
        with open("short_character_anno.txt", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines()
            new_annos += short_character_anno
    # Source 2: transcribed long audio segments
    if os.path.exists("./long_character_anno.txt"):
        with open("./long_character_anno.txt", 'r', encoding='utf-8') as f:
            long_character_anno = f.readlines()
            new_annos += long_character_anno

    # Get all speaker names
    speakers = []
    for line in new_annos:
        path, speaker, text = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."
    
    # STEP 1: modify config file
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # assign ids to new speakers
    speaker2id = {}
    for i, speaker in enumerate(speakers):
        speaker2id[speaker] = i
    # modify n_speakers
    hps['data']["n_speakers"] = len(speakers)
    # overwrite speaker names
    hps['speakers'] = speaker2id
    hps['train']['log_interval'] = 10
    hps['train']['eval_interval'] = 50
    hps['train']['batch_size'] = 8
    hps['data']['training_files'] = "final_annotation_train.txt"
    hps['data']['validation_files'] = "final_annotation_val.txt"
    # save modified config
    with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2)

    # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
    import text

    cleaned_new_annos = []
    for i, line in enumerate(new_annos):
        path, speaker, txt = line.split("|")
        if len(txt) > 150:
            continue
        cleaned_text = fix_text(txt)
        # Text already cleaned
        # cleaned_text = text._clean_text(cleaned_text, hps['data']['text_cleaners'])
        cleaned_text = phoneme_text(cleaned_text)
        cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
        cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

    final_annos = cleaned_new_annos
    # save annotation file
    with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
        for line in final_annos:
            f.write(line)
    # save annotation file for validation
    with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
        for line in cleaned_new_annos:
            f.write(line)
    print("Finished")
