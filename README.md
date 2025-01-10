# ImageGeneration
ML@Purdue
-Take 1000 images per class from mnist numbers
	Here we will just take 1000 images from 0 through 9 from mnist numbers.

-Train Gan on that subset of the data
	Using that 1000 per class dataset we will train a GAN. For optimised images for each class in the data we train the gan individually for each digit (we have already tried this approach and demonstrated drastically improved results from our GAN).

-Generate 1000 images per class from the Gan
	We will generate these images into a labeled drive to be used in CNN training.

-Train a CNN on the 1000 image per class subset of Mnist
	We will likely try to use some sort of preexisting CNN backbone here like YOLO or vgg16 to make sure that CNN performance doesnt bring anything new or unexpected to the table.

-Train a CNN on the 1000 generated per class images
	Similar process to above except we use the generated images.

-Train a CNN on the whole mnist numbers dataset
	Just train the CNN using the base mnist numbers dataset.

-Compare results 
	See if there are any significant differences between the performance of the 3 CNN’s on a validation set. Especially look to see if the CNN trained on synthetic data was able to pick up any out of dataset information due to the noise introduced during the generation of the data by the GAN.

-Repeat for other types of image classes
	We predict that any disparities between the 1000 per class real and synthetic trained CNNs will become further pronounced as the complexity of the images increases (due to greater possibility of out of dataset information existing in the first place).
