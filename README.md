Introduction

Medical imaging plays a central role in modern clinical diagnosis, decision-making, and surgical planning. Modalities such as Magnetic Resonance Imaging (MRI), Computed Tomography (CT), Ultrasound (US), and X-rays provide invaluable visual insights into the human body. However, the quality of medical images is often compromised due to noise, poor contrast, illumination variance, and resolution limitations introduced during acquisition, transmission, or storage. Poor-quality images not only make diagnosis difficult but may also lead to misinterpretation and misdiagnosis. Enhancing the quality of such medical images is thus a necessity, not just for better visualization but for clinical safety. Enhancing medical images using computational techniques has been an evolving research area in digital image processing, but many traditional tools require domain expertise, command-line skills, or expensive licenses.

Motivation

The core motivation for this project stems from the need to build a simple yet powerful tool that empowers healthcare practitioners, researchers, and students to enhance and analyze medical images without coding or expensive software. The goal was to democratize medical image processing using a user-friendly web-based interface, powered by Streamlit and Python libraries like Scikit-Image, OpenCV, and NumPy. A web interface allows real-time interaction, visual comparison, and intuitive control, making it ideal for educational settings and preliminary analysis workflows. Integrating modern enhancement techniques such as adaptive histogram equalization, denoising, edge detection, segmentation, and advanced filtering techniques ensures that the tool addresses both basic and advanced use cases.

Objectives

The objectives of the project are clearly defined as follows:
•	To design and develop a web-based tool for uploading and enhancing medical images.
•	To implement a suite of image processing techniques (contrast, brightness, gamma, filters, segmentation).
•	To integrate evaluation metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
•	To allow real-time visualization of original vs. enhanced images and histograms.
•	To enable download of the processed medical images in standard formats.

