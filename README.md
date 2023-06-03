# Anomaly Detection

Task1: Segmentation And Annotation

![alt text](data/1-1.png)

Task2: Propose An Anomaly Detection Model

Step1: Generate new samples using OK images using HistAuGAN
![alt text](data/2-0.png)


Step2: Fine-Tune teacher model on OK images
![alt text](data/2-1.png)


Step3: Use Teacher model to teach student model by minimising the teacher student extracted features for OK images. Get anomaly map by calculating the similarity between student and teacher extracted features for sample images.
![alt text](data/2-2.png)


Task3: Anomaly Detection Given a Single Good Template

Working colab notebook(Working_Demo_anomaly_detection.ipynb) available for using different files 
