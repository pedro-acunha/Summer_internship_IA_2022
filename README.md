# Summer_internship_IA_2022
Collection of material used during the IA Summer Internship 2022 for the project: **Automatic classification of galaxies using the Galaxy Zoo data and supervised learning.**

**Supervisors**: *Pedro Cunha & Ana Paulino-Afonso*

# Detailed Plan:
## Task 1: Exploring Galaxy Zoo data
The Galaxy Zoo project provided the classification results in the following website:
https://data.galaxyzoo.org/

The first thing to do is download the data. You will get the dataset from Galaxy Zoo 2 (https://zenodo.org/record/3565489#.YsglD9JByV6).
Read the description page carefully alongside with this paper: https://doi.org/10.1093/mnras/stt1458. It is important to cross-reference the images with the classification from Galaxy Zoo. You can do it by using the ObjID. The class you will consider for the classification is the “gz2_class”.
My recommendation is for you to identify the classes in the dataset and select a random
sample of sources with that label (e.g. 2,000 galaxies classified as Er, etc). You are free
to choose the number of classes you want to use (e.g, 2 for a binary classification, or all
of them). Remember that the number of chosen classes will increase the size of the
data set and the computation processing time.
At the end, you should have a main folder with subfolders that corresponds to the
classes of the galaxies. This will be helpful for later!

## Task 2: Preparing the pipeline
After the data processing task, you need to start building the pipeline.
Here I propose you check the following examples:
- https://www.tensorflow.org/tutorials/keras/classification
- https://towardsdatascience.com/create-image-classification-models-with-tensorflo
w-in-10-minutes-d0caef7ca011
- https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95
This is an experimental task, which means you will do a lot of things by trial and error. It
is important you understand the different steps and how they are relevant for your data
set.

## Task 3: Testing your model
After you have your model ready, it is time for evaluation, since we are doing a
supervised task. This task is actually pretty linked with the previous one. You should
build at least 2 models: (1) Baseline: this should be a simple one for comparison and to
understand how complexity help the problem in hand; (2) CNN: Taking into
consideration the model (1), you can try to add more layers to the deep learning model,
in particular 2D convolutional layers. You are encouraged to test multiple models and
achieve the best result possible.

Have fun!
