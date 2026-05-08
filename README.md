# SGP4SR (AAAI'26)

Sequential recommendation model incorporating semantic and graph pooling.

## Project Structure

```
├── data/                    # Data loading and transformation code
│   ├── dataloader.py        # Custom dataloader with transforms
│   ├── dataset.py           # SGPDataset - loads embeddings and interactions
│   └── transform.py         # Data augmentation transforms
├── dataset/                 # Processed dataset (train/valid/test .inter files)
│   ├── baby/
│   └── office/
├── data_raw/                # Raw Amazon data (before preprocessing)
│   ├── baby/
│   └── office/
├── processed/               # Preprocessed features (image/text embeddings)
│   ├── baby/
│   └── office/
├── scripts/
│   └── preprocess_amazon_old.py  # Raw data → dataset preprocessing
├── run.py                   # Main training script
├── sgp.py                   # SGP model implementation
├── model_utils.py           # Model utilities
├── SGP4SR.yaml              # Model config
├── run.yaml                 # Training config
└── README.md
```

## Data Pipeline

### 1. Download Raw Data from Amazon

Download review, metadata, and image features from [Stanford SNAP](http://snap.stanford.edu/data/amazon/productGraph/):

```powershell
# Create directories
mkdir data_raw\baby data_raw\office

# Baby category
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby.json.gz" -o "data_raw\baby\reviews_Baby.json.gz"
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz" -o "data_raw\baby\meta_Baby.json.gz"
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Baby.b" -o "data_raw\baby\image_features_Baby.b"

# Office Products category
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products.json.gz" -o "data_raw\office\reviews_Office_Products.json.gz"
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz" -o "data_raw\office\meta_Office_Products.json.gz"
curl.exe -L "http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Office_Products.b" -o "data_raw\office\image_features_Office_Products.b"
```

### 2. Preprocess Data

Process raw data: filter users/items (k-core), build sequences, extract embeddings:

```powershell
# Baby dataset
python scripts/preprocess_amazon_old.py `
  --name baby `
  --reviews data_raw\baby\reviews_Baby.json.gz `
  --meta data_raw\baby\meta_Baby.json.gz `
  --image-features data_raw\baby\image_features_Baby.b `
  --out-dataset dataset\baby `
  --out-processed processed\baby `
  --min-user 5 `
  --min-item 5

# Office dataset
python scripts/preprocess_amazon_old.py `
  --name office `
  --reviews data_raw\office\reviews_Office_Products.json.gz `
  --meta data_raw\office\meta_Office_Products.json.gz `
  --image-features data_raw\office\image_features_Office_Products.b `
  --out-dataset dataset\office `
  --out-processed processed\office `
  --min-user 5 `
  --min-item 5
```

Preprocessing output:
- `dataset/{name}/{name}.{train,valid,test}.inter` – Sequential interactions
- `processed/{name}/image_features.npy` – Image embeddings (4096-dim)
- `processed/{name}/text_features.npy` – Text embeddings (384-dim, if available)

## Training

```bash
python run.py -d baby
python run.py -d office
```

Config files:
- `SGP4SR.yaml` – Model hyperparameters
- `run.yaml` – Data & training configs
