# Anomaly Detection

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains code and data for an anomaly detection project, which involves segmentation and annotation, proposing an anomaly detection model, and detecting anomalies given a single good template. The project utilizes various techniques, including HistAuGAN for generating new samples and teacher-student models for feature extraction.

## Segmentation And Annotation
The first step of the project is segmentation and annotation. An example image from this phase is shown below:

![Segmentation and Annotation](data/1-1.png)

## Propose An Anomaly Detection Model

### 1. Generate New Samples Using HistAuGAN
HistAuGAN is employed to generate new samples using OK (normal) images. The generated samples can be accessed [here](https://drive.google.com/drive/folders/1VJfZKvlc-h2P9r0uLWA34kFx21atnXcA?usp=sharing). An example of the generated sample is shown below:

![Generated Sample](data/2-0.png)

### 2. Fine-Tune Teacher Model on OK Images
The teacher model is fine-tuned using OK images. An illustration of this step is depicted below:

![Fine-Tune Teacher Model](data/2-1.png)

### 3. Use Teacher Model to Teach Student Model
The student model is taught by the teacher model using OK images. This is achieved by minimizing the similarity between teacher and student extracted features for the OK images. Anomaly maps are obtained by calculating the similarity between the student and teacher extracted features for sample images. An example of the anomaly map is shown below:

![Anomaly Map](data/2-2.png)

## Anomaly Detection Given a Single Good Template

### 1. Generate New Samples Using OK Images and Good Template
Similar to the previous step, new samples are generated using OK images and a good template, utilizing HistAuGAN. The generated samples are available [here](https://drive.google.com/drive/folders/1VJfZKvlc-h2P9r0uLWA34kFx21atnXcA?usp=sharing). An example of the generated sample is shown below:

![Generated Sample with Template](data/3-0.png)

### 2. Fine-Tune Teacher Model on OK Images
The teacher model is again fine-tuned, this time using OK images and the good template.

![Fine-Tune Teacher Model](data/2-1.png)

### 3. Use Teacher Model to Teach Student Model
Similar to the previous approach, the teacher model is used to teach the student model by minimizing the similarity between teacher and student extracted features for OK images. Anomaly maps are then obtained for the sample images. An example of the anomaly map is shown below:

![Anomaly Map](data/2-2.png)

## Comparison Between No-Template and Template Anomaly Detection

The repository includes a comparison between no-template and template anomaly detection results. Here are two sample images, one for each case:

| No-Template | Template |
| --- | --- |
| ![No-Template](data/scratch_1_1.png) | ![Template](data/scratch_2_1.png) |

## Colab Notebook

A [Colab Notebook](https://github.com/TapasKumarDutta1/anomaly-detection/blob/main/Working_Demo_anomaly_detection.ipynb) is provided for utilizing different files and experimenting with the anomaly detection process.

For any questions or issues related to the project, feel free to open an issue in the repository.

*Note: This readme provides an overview of the project and its processes. Detailed explanations and code implementation can be found in the respective files and notebooks.*


## Citation

```markdown
@inproceedings{rottshaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}

@inproceedings{HistAuGAN,
  author = {Wagner, S. J., Khalili, N., Sharma, R., Boxberg, M., Marr, C., de Back, W., Peng, T.},
  booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2021},
  title = {Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations},
  year = {2021}
}
