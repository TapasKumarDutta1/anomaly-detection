# Anomaly Detection

1 Segmentation And Annotation
![alt text](data/1-1.png)

2 Propose An Anomaly Detection Model

    1 Generate new samples using OK images using HistAuGAN
    ![alt text](data/2-0.png)
    
    2 Fine-Tune teacher model on OK images
    ![alt text](data/2-1.png)
    
    3 Use Teacher model to teach student model by minimising the teacher student extracted features for OK images. Get anomaly map by        calculating the similarity between student and teacher extracted features for sample images.
    ![alt text](data/2-2.png)

3 Anomaly Detection Given a Single Good Template
    
    1 Generate new samples using OK images and Good Template using HistAuGAN
    ![alt text](data/3-0.png)
    
    2 Fine-Tune teacher model on OK images
    ![alt text](data/2-1.png)
    
    3 Use Teacher model to teach student model by minimising the teacher student extracted features for OK images. Get anomaly map by        calculating the similarity between student and teacher extracted features for sample images.
    ![alt text](data/2-2.png)


No-Template|Template
 --- | ---
![alt text](data/scratch_1_1.png) | ![alt text](data/scratch_2_1.png)

Working colab notebook(Working_Demo_anomaly_detection.ipynb) available for using different files 
