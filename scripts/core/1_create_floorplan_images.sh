# run from root directory
# bash scripts/core/1_create_floorplan_images.sh

input_directory='./data/geodata'
output_directory='./data/images/floorplan'
module='floor_plan'
step='generate_images'

python ./main/core.py --module=floor_plan --step=generate_images --input_directory=./data/geodata --output_directory=./data/images/floorplan
