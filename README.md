## Anomaly Detection
Clone the repository and unzip all files in the same directory

Task 1: Segmentation of templates from images
Steps:
  1. Create directory segmented to store segmented templates within the images
  2. Extract one template from any image manually and use matchTemplates of Task1.py to extract most similar templates
  extracted samples: https://drive.google.com/drive/folders/13u3WkebVPQaxyiZAzPoTmUjDkSu9Lk1e?usp=sharing

Task 2: Anomaly Detection within images(without GAN augmentation)
Steps:
  1. Install pytorch_lightning==1.9.4
  2. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path


Task 2: Anomaly Detection within images(with GAN augmentation)
Steps:
  1. 
  1. Install pytorch_lightning==1.9.4
  2. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path


## Things i tried that did not work

  1. Augmentation of single image using SinGAN(generated images had their structure damaged)
  2. Blockwise finetunning and Standard finetunning GAN with augmentation model on template augmented data  
  
## Things that gave small improvement(needs further experimentation)

  1. Training model with Horizontal/ Vertical/ ColorJitter augmented data and using these for test time augmentation
