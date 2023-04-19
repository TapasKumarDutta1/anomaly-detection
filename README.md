# Anomaly Detection
Clone the repository and unzip all files in the same directory

## Task 1: Segmentation of templates from images

  1. Create directory segmented to store segmented templates within the images
  2. Extract one template from any image manually and use matchTemplates of Task1.py to extract most similar templates
  extracted samples: https://drive.google.com/drive/folders/13u3WkebVPQaxyiZAzPoTmUjDkSu9Lk1e?usp=sharing

## Task 2: Anomaly Detection within images(without GAN augmentation)

  1. Install pytorch_lightning==1.9.4
  2. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path
  Samples with anomaly map:


## Task 2: Anomaly Detection within images(with GAN augmentation)

  1. Augment Ok images using HistAuGAN combine the GAN augmented images with Ok images for training
  ** Use weight 0.1 for GAN images during training. Samples: 
  3. Install pytorch_lightning==1.9.4
  4. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path
  Samples with anomaly map:

## Task 3: Define Defects Type and Give a Quantitative Evaluation of the Defect Level

  Defect Types and quantitative evaluation
  ### 1. Blurry Text:
          Definition: Texts are blurry
          Condition: Anomaly map covers text region
  ### 2. Structural Defect:
          Definition: Structures are malformed
          Condition: Anomaly map covers structure edges
  ### 3. Color/Scratch Defect:
          Definition: Color of region different from surrounding
          Condition: Anomaly map present within structure 
          
## Things i tried that did not work

  1. Augmentation of single image using SinGAN(generated images had their structure damaged)
  2. Blockwise finetunning and Standard finetunning GAN with augmentation model on template augmented data  
  
## Things that gave small improvement(needs further experimentation)

  1. Training model with Horizontal/ Vertical/ ColorJitter augmented data and using these for test time augmentation
