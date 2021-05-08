# run from root directory
# bash scripts/core/2_create_floorplan_dataset.sh

input_directory='./data/images/floorplan'
output_directory='./data/datasets/floorplan'
module='floor_plan
step='generate_dataset'

python ./main/core.py \
  --module=$module \
  --step=$step \
  --input_directory=$input_directory \
  --output_directory=$output_directory
