# -Distributed-Deep-learning-on-Caltech-UCSD-Birds-200-dataset-

Caltech-UCSD Birds 200 (CUB-200) is an image dataset with photos of 200 bird species (mostly North American ). For testing phase, I considered 5 birds and used VGG-16 CNN Architecture and with proper pre-processing and augmentation, I reached accuracy of 98.12 percent. The same, I implemented on distributed spark-environment on cluster and by using same architecture we got similar accuracy with way less execution time compared to the original python script on single system.
