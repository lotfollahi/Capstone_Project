

1) What are you trying to do? Articulate your objectives using absolutely no jargon (i.e. as if you were explaining to a salesperson, executive, or recruiter).

The goal is tissue classification in histological samples to generate a chemical map that can be used by a pathologist to identify the spatial distribution of tissue types within the sample. 

2) How has this problem been solved before? If you feel like you are addressing a novel issue, what similar problems have been solved, and how are you borrowing from those?

Tissue classification is achieved in most infrared imaging studies using pixel-level methods, including unsupervised techniques such as K-means clustering or hierarchical cluster analysis (HCA), and supervised techniques such as Bayesian classification, random forests, artificial neural networks (ANNs), kernel classifiers such as support vector machines (SVMs), and linear discriminant classifiers. Fourier transform infrared (FTIR) spectroscopic data contains an abundance of spatial information that is often unused due to difficulty identifying useful features. Spatial information has been utilized in multi-modal applications involving FTIR and traditional histology, since spatial features are more clearly understood in standard color images. Spatial information has also been used as a post-classification step on the classified output. However, these approaches are sequential to spectral analysis and do not take advantage of the spectral-spatial relationships within the IR data set.

3) What is new about your approach, why do you think it will be successful?

Convolutional neural networks (CNNs) are the current method of choice for image analysis, since they exploit spatial features by enforcing local patterns within the image. In addition to extracting spatial correlations between pixels, CNNs can be implemented for hyperspectral images, extracting correlations across the entire spectrum for a given pixel. CNNs have therefore become an effective machine learning tool for image classification tasks. 



4)	Who cares? If you're successful, what will the impact be?
Current methods for cancer detection rely on tissue biopsy, chemical labeling/staining, and examination of the tissue by a pathologist. This work demonstrates the application and efficiency of deep learning algorithms in improving the diagnostic techniques in clinical and research activities related to cancer without any need to chemical staining. 


5) How will you present your work?
I will use slide show. The final visuals are painted biopsies with different colors for different cellular biomolecules.


6) What are your data sources? What is the size of your dataset, and what is your storage format?

Tissue samples are imaged using FTIR spectroscopy. The dataset comprised of 680 IR- spectra (680 bands) which is 680 features for each pixel. 


7)What are potential problems with your capstone, and what have you done to mitigate these problems?

The size of data is very large that might slow done the process. I may also need to use dimensionality reduction to reduce size of the data which in that case choosing right method (PCA, NMF, UVD, SVD) are challenging for me. 

8) What is the next thing you need to work on?

Next step is to transfer data from university cloud to AWS-S3 and understanding the data.  




