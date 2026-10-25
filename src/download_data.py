import gdown
from pathlib import Path

FILE_ID = "1C7tcEdnRIlxwIsFnsN6F0jkpU1FRlieS"
DEST_PATH = Path(r"c:\Users\khole\OneDrive\Desktop\EMIpredict AI\data\raw\emi_prediction_dataset.csv")

def download_data():
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading dataset from Google Drive: {url}")
    gdown.download(url=url, output=str(DEST_PATH), quiet=False)
    print(f"Dataset successfully saved to: {DEST_PATH}")

if __name__ == "__main__":
    download_data()
