# FuseMine:Robust Multi-Modal Compound-Protein Interaction Prediction via Differential Attention Feature Mining

## 📦 Dependencies

- `python==3.9.21`
- `pytorch==2.2.1`
- `pytorch-cuda==12.1`
- `pytorch-lightning==2.5.1`
- `dgl==2.4.0+cu121`
- `dgllife==0.3.2`
- `easydict==1.13`
- `einops==0.8.1`
- `fair-esm==2.0.0`
## 🚀 How to Run
Step 1: Generate ESM and Chemberta Embeddings
```bash
python preprocess/get_embeddings.py
```
This will generate and save protein and compound embeddings required for training.

Step 2: Train the Model
Use train.py with appropriate arguments:
```bash
python train.py \
  --outname <output_folder_name> \
  --data <dataset_name> \
  --num_worker <num_dataloader_workers> \
  --type <training_type> \
  [--gpus <gpu_ids>] 
```
Example Runs:
```bash
# Train on BindingDB with cluster-based splitting using specific GPUs
python train.py --outname bindingdb_model --data bindingdb --num_worker 0 --type cluster --gpus 5,6
```
