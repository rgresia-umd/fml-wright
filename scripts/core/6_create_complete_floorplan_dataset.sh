# run from root directory
# bash scripts/core/6_create_complete_floorplan_dataset.sh

input_directory='./data/images/complete_floorplan'
output_directory='./data/datasets/complete_floorplan'
module='complete_floorplan'
step='generate_dataset'

python ./main/core.py --module=complete_floorplan --step=generate_dataset --input_directory=./data/images/complete_floorplan --output_directory=./data/datasets/complete_floorplan
