# Anomaly Detection

1. Segmentation And Annotation
![alt text](data/1-1.png)

2. Propose An Anomaly Detection Model
[Output Anomaly Maps](https://drive.google.com/drive/folders/1Q6xTlgtAseS8PYk3Mc4VIpS7Wt-E757J?usp=sharing)

    1. Generate new samples using OK images using HistAuGAN
[Generated Samples](https://drive.google.com/drive/folders/1VJfZKvlc-h2P9r0uLWA34kFx21atnXcA?usp=sharing)
    ![alt text](data/2-0.png)
    
    2. Fine-Tune teacher model on OK images
    ![alt text](data/2-1.png)
    
    3. Use Teacher model to teach student model by minimising the teacher student extracted features for OK images. Get anomaly map by        calculating the similarity between student and teacher extracted features for sample images.
    ![alt text](data/2-2.png)

3. Anomaly Detection Given a Single Good Template
[Output Anomaly Maps](https://drive.google.com/drive/folders/18FoeMpwZ98pnrnw44jxtWnmPgg3wd6rG?usp=sharing )
    
    1. Generate new samples using OK images and Good Template using HistAuGAN 
[Generated Samples](https://drive.google.com/drive/folders/1VJfZKvlc-h2P9r0uLWA34kFx21atnXcA?usp=sharing)
    ![alt text](data/3-0.png)
    
    2. Fine-Tune teacher model on OK images
    ![alt text](data/2-1.png)
    
    3. Use Teacher model to teach student model by minimising the teacher student extracted features for OK images. Get anomaly map by        calculating the similarity between student and teacher extracted features for sample images.
    ![alt text](data/2-2.png)


No-Template|Template
 --- | ---
![alt text](data/scratch_1_1.png) | ![alt text](data/scratch_2_1.png)

[Working colab Notebook](https://github.com/TapasKumarDutta1/anomaly-detection/blob/main/Working_Demo_anomaly_detection.ipynb) available for using different files 
