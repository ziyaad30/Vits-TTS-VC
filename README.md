# Vits-TTS-VC

## Tested on Python 3.8.17

1. git clone https://github.com/ziyaad30/Vits-TTS-VC.git
2. cd Vits-TTS-VC
3. pip install -r requirements.txt
4. cd monotonic_align
5. python setup.py build_ext --inplace
6. cd ..
7. Download [DeepPhonemizer](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt) and place the model inside Vits-TTS-VC directory.
