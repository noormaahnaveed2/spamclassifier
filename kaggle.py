def kaggle():
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sahideseker/spam-mail-classifier-dataset")

print("Path to dataset files:", path)