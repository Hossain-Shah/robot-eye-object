# SSD-based Object Detection in PyTorch

## Environment setup:
 - conda activate object_detection_nano (GPU device 3090)

(*** Please create a new conda environment by "conda create -n 'environment_name'" and follow conda_requirements.txt and input "conda install -r conda_requirements.txt" command if you are not in the GPU device 3090 ***)

## Dataset preparation:
 - For existing 600 classes data of "Open images", only "Data collection" step is enough before "Model training". But if custom data classes need to be added, "Data preprocessing" and "Dataset update" steps are also compulsory. Here, for present scenario, 12 classes are included in the detection data-set where 11 classes
are gathered from the mentioned data-set and Onion class is processed from self processing and annotation since it was not available in the mentioned data-set. Annotation is in “Open images”format. Total 1126 images are used where train, validation and test ratio is:
 - Train: 899
 - Validation: 65
 - Test: 162

## Data collection:
 - python open_images_downloader.py  --max-images=Input your required number of images per class here --class-names "Input your class names here"  --data=data/Input your data directory name 

## Data preprocessing:(Only for custom data classes)
 - Use VGG image annotator(https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) for data annotation and save it in coco json format.
 - python coco2voc.py --ann_file coco_json_annotation_file --output_dir preprocessed_data/annotations
 - Copy your target images in preprocessed_data 
 - python pascal_voc_xml_to_csv_conversion.py

(*** Please make sure that preprocessed_data and its annotations folder is fully empty before each annotation process ***)

## Dataset update:(Only for custom data classes)
 - Add images to existing data folder's sub-folder(got from 2nd step(Data collection)) into train, validation, test folder accordingly.
 - Add any label_name/id in class-descriptions-boxable.csv file's 1st column, but must be unique for each class. In 2nd column, input class_name(give relevant name). 
 - In bbox annotation csv files, add information accordingly. 
    * 1st column, "ImageID" must be similar to actual jpg images name
    * 2nd column, "Source" can be left blank.
    * 3rd column, "LabelName" must be similar to "class-descriptions-boxable.csv" file's 1st column value.
    * 4th column, "Confidence" must be set as 1 as previous rows.
    * 5th column, "XMin" must be equal to "labels_crop_name_preprocessed_data.csv" file's 5th column value(xmin) divided by 2nd column value(width).
    * 6th column, "XMax" must be equal to "labels_crop_name_preprocessed_data.csv" file's 7th column value(xmax) divided by 2nd column value(width).
    * 7th column, "YMin" must be equal to "labels_crop_name_preprocessed_data.csv" file's 6th column value(ymin) divided by 3rd column value(height).
    * 8th column, "YMax" must be equal to "labels_crop_name_preprocessed_data.csv" file's 8th column value(ymax) divided by 3rd column value(height).
    * 9th, 10th, 11th, 12th, 13th column values can be left blank.
    * 14th column, "id" must be similar to 3rd column, "LabelName" values.
    * 15th column, "ClassName" must be similar to "class-descriptions-boxable.csv" file's 2nd column value.      

## Model training:
Model training: Model is trained using SSD-Mobilenet-v1 detection process
applying 300 epochs and 256 batches. Parameters used are:
    - Resolution: (300, 300)
    - Learning rate: 0.01
    - Momentum: 0.9
    - Weight-decay: 5e-4
    - Gamma: 0.1
 - python train_ssd.py --data=data/Input your data directory name --model-dir=models/Input your model directory name --batch-size=Input your required batch size --epochs=Input your required number of epochs

## Model inference:
 - python run_ssd_example.py model_type model_checkpoint_path label_path testing_source_path

## Onnx conversion:
 - python onnx_export.py --model-dir=models/Input your model directory name

## Model evaluation:
 - python eval_ssd.py --net model_type --model model_checkpoint_path --dataset_type dataset_type --dataset dataset_path --label_file label_path --eval_dir evaluation_path

## Model performance graph:
The best performance recorded is a loss amount of 3.88 after
crossing 280 epochs.

 - python result.py 

## Inference in Jetson NANO:
 - 1st entering into recognition project inference code directory 
  * cd /home/nsl-nano/jetson-inference/build/aarch64/bin
 - python3 detectnet-camera.py --model=networks/Input your model directory name/ssd-mobilenet.onnx --class_labels=networks/Input your model directory name/labels.txt --input_blob=input_0 --output_cvg=scores --output_bbox=boxes --headless --input-codec=raw --width=1280 --height=720 --camera=/dev/video0

## Project setup for Jetson NANO:
 - If project has to be setup again for any system loss, then follow these steps:
  * sudo apt-get update
  * sudo apt-get install dialog
  * sudo apt-get install git cmake libpython3-dev python3-numpy
  * git clone --recursive https://github.com/dusty-nv/jetson-inference
  * cd jetson-inference
  * mkdir build
  * cd build
  * cmake ../
  * make
  * sudo make install
  * sudo ldconfig 


## Reference:
> For clarification, you may go through "https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md"
