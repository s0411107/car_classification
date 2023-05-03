model is to big can not put on github

you may need to check setting and install some package
python version 3.10.0
pip install tensorflow
pip install pillow


put the test folder inside the DATA folder if you just want to copy the command to use , 
otherwise put your own path in --image_folder_path to use 

use 
python test_model.py --model_path model-resnet50-final.h5 --image_folder_path ./DATA/test --output_file_path output.txt

to run program


