python scripts/export_celeba_for_repa.py
cd REPA/preprocessing
python dataset_tools.py encode \
    --source ../../data/celeba256 \
    --dest ../../data/celeba256/vae-sd \
    --model-url stabilityai/sd-vae-ft-mse
cd ../..
bash scripts/run_celeba_baseline.sh
bash scripts/run_celeba_repa.sh
