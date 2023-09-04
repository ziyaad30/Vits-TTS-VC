import os
import requests
from tqdm import tqdm

'''
def download(url: str, dest_folder: str, new_filename: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text)) 
    
    new_file_path = os.path.join(dest_folder, new_filename)    
    os.rename(file_path, new_file_path)    
'''

def download(url: str, dest_folder: str, new_filename: str, chunk_size=1024):
    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    
    with open(file_path, 'wb') as file, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    
    new_file_path = os.path.join(dest_folder, new_filename)    
    os.rename(file_path, new_file_path) 

if not os.path.isfile("pretrained_models/D_0.pth"):
    print("Downloading D_0.pth")
    download("https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth", dest_folder="pretrained_models", new_filename="D_latest.pth")
    
if not os.path.isfile("pretrained_models/G_0.pth"):
    print("Downloading G_0.pth")
    download("https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth", dest_folder="pretrained_models", new_filename="G_latest.pth")
