# Anomaly Detection
Clone the repository and unzip all files in the same directory

## Task 1: Segmentation of templates from images

  1. Create directory segmented to store segmented templates within the images
  2. Extract one template from any image manually and use matchTemplates of Task1.py to extract most similar templates
  extracted samples: https://drive.google.com/drive/folders/13u3WkebVPQaxyiZAzPoTmUjDkSu9Lk1e?usp=sharing

## Task 2: Anomaly Detection within images(without GAN augmentation)

  1. Install pytorch_lightning==1.9.4
  2. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path
  Samples with anomaly map: https://drive.google.com/drive/folders/1FkrorrVT8h_XQwq50Qr21TcmuBDabdXB?usp=sharing


## Task 2: Anomaly Detection within images(with GAN augmentation)

  1. Augment Ok images using HistAuGAN combine the GAN augmented images with Ok images for training
  ** Use weight 0.1 for GAN images during training. Samples: https://drive.google.com/drive/folders/1-cMRgjqylpox1SWk8xg1MO5i_PD7aRnl?usp=sharing
  3. Install pytorch_lightning==1.9.4
  4. Run Task2.py and specify the path to Ok images in dataset_path and path to store logs in project_path
  Samples with anomaly map: https://drive.google.com/drive/folders/1biDmpL5HiqCzcXJ5hIOmNem2TPQKQLiw?usp=sharing

## Task 3: Define Defects Type and Give a Quantitative Evaluation of the Defect Level

  ### 1. Blurry Text:
          Definition: Texts are blurry
          Condition: Anomaly map covers text region
  ### 2. Structural Defect:
          Definition: Structures are malformed
          Condition: Anomaly map covers structure edges
  ### 3. Color/Scratch Defect:
          Definition: Color of region different from surrounding
          Condition: Anomaly map present within structure 
  
  ### Quantitative Evaluation: 
  
  Area covered by observable largest defect 
          
## Task 4: Anomaly Detection Given a Single Good Template

   1. Augment images with HistAuGAN setting Ok images as one type and template as another 
   Experimented with probability of selecting Template image
   
   1%: https://drive.google.com/drive/folders/1LhUR-SzNWMPp0viIopxOqoSf7ZXLhoI_?usp=sharing
   10%: https://drive.google.com/drive/folders/1bpsbeyorbLP2kPe3qjxX0-rup6L08o4S?usp=share_link
   50%: https://drive.google.com/drive/folders/19FTEdGwOk8gM9fQv1qHOINaIp0qk6DVP?usp=sharing
   
   2. For content vector from Ok images and its domain use 'generate_hist_augs function to convert them as much as possible to Template image
   3. Combine the GAN augmented images(10% probability of Template and OK augmented images) with Ok images and previously generated images for training and run Task2.py
   Results: https://drive.google.com/drive/folders/1geQyfDzV7H_287PRxbxb4shNSI8dbxcO?usp=sharing
          
## Things i tried that did not work

  1. Augmentation of single image using SinGAN(generated images had their structure damaged) Samples: https://drive.google.com/drive/folders/1gnlYlt-ZiLR67LZVC3vMyVinxIsMwI33?usp=sharing
  2. Blockwise finetunning and Standard finetunning GAN with augmentation model on template augmented data 
  
## Things that gave small improvement(needs further experimentation)

  1. Training model with Horizontal/ Vertical/ ColorJitter augmented data and using these for test time augmentation
