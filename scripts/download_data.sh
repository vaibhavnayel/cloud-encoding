mkdir data
cd data
aria2c -x12 https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
unzip dataset_small_v1.1.zip

if [ ! -f "ShapeNet/metadata.yaml" ]; then
    cp metadata.yaml ShapeNet/metadata.yaml
fi
