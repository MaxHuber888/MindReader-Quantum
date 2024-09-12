---
title: MindReader Quantum
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
short_description: Classical-to-Quantum CNN (with transfer learning)
---

# MindReader Quantum

![](./mindreader-quantum.JPG)

In this project, developed during the 2023 MIT IQuHack, my team tackled the IBM/Covalent challenge to leverage quantum computing for a service benefiting everyday users. We created a Hybrid Classical-to-Quantum Neural Network designed to detect dementia severity from brain MRI scans, providing an innovative solution to a pressing healthcare need.

Motivation
The hybrid network structure was chosen to combine the strengths of both classical and quantum computing. By using a pretrained classical Convolutional Neural Network (CNN) to extract rich feature vectors from MRI images, we could take advantage of transfer learning. This approach minimizes the need for large amounts of quantum training data, as the classical CNN already provides generalized, high-level representations of the input data.

We then fine-tuned the model using a quantum neural network head. Quantum neural networks (QNNs) offer powerful advantages, especially in terms of parallelization, where quantum computing excels. This allows the model to process complex patterns in the data more efficiently than classical counterparts, particularly when fine-tuning to specific tasks such as detecting varying levels of dementia severity.

Implementation and Accessibility
Our hybrid network architecture was trained and evaluated using both classical and quantum computing resources, allowing us to push the boundaries of current AI applications. The next step involved connecting this backend to a React-based frontend, aiming to make the service easily accessible to users via their mobile devices.

Our ultimate goal was to make advanced medical imaging analysis more accessible, allowing non-expert users to understand their MRI results and receive early dementia screening from their phones, powered by cutting-edge quantum technology.

## Acknowledgments

We would like to express our gratitude to the open-source community and the developers of the tools and technologies that made this project possible. Additionally, we extend our thanks to the healthcare professionals and researchers working tirelessly to improve the lives of those affected by Alzheimer's and dementia.
