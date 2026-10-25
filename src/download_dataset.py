import urllib.request
import time
import os

URL = "https://drive.google.com/uc?export=download&id=1C7tcEdnRIlxwIsFnsN6F0jkpU1FRlieS"
OUTPUT_FILE = r"c:\Users\khole\OneDrive\Desktop\EMIpredict AI\data\raw\emi_prediction_dataset.csv"

def download():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    print(f"Connecting to Google Drive...")
    start_time = time.time()
    
    with urllib.request.urlopen(req) as resp, open(OUTPUT_FILE, "wb") as out_file:
        chunk_size = 1024 * 512  # 512 KB
        downloaded = 0
        last_print = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_print >= 5 * 1024 * 1024:
                elapsed = time.time() - start_time
                speed = (downloaded / (1024 * 1024)) / (elapsed + 1e-5)
                print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB ({speed:.2f} MB/s)", flush=True)
                last_print = downloaded

    total_mb = downloaded / (1024 * 1024)
    print(f"Download complete: {OUTPUT_FILE} ({total_mb:.2f} MB in {time.time()-start_time:.1f}s)", flush=True)

if __name__ == "__main__":
    download()
