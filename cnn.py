# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# 32 are the number of filters/feature detectors and 3x3 is the size.
# It is most common practice to use 3x3 feature detectors.
# Use bigger size images than (64,64) if using a GPU
# This input_shape is for a tensorflow backend.
# For A Theano backend, use (3,64,64)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Size of the pooling matrix is 2x2 in majority cases.


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# 'adam' here represents Stochastic Gradient Descent algorithm.

# Part 2 - Fitting the CNN to the images

# Image Augmentation is a technique that allows us to enrich our dataset without adding more images.
# Therefore, allows us to get good results and prevent overfitting even with small amount of images.


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# The parts above and below are Image Augmentation.
# The rescale parameter insures we have pixel values between 0 and 1.
# We are creating objects of ImageDataGenerator class to apply augmentation to.

test_datagen = ImageDataGenerator(rescale = 1./255)


# Now apply the Image augmentation with those objects above.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# To improve accuracy of this model, choose a higher target_size for your images.
# But this is increase the computation and training time of course.

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# 8000 because we have 8000 images in training set.
# 2000 because we have 2000 images in test set.

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) # Convert Image to a 3D Array.
test_image = np.expand_dims(test_image, axis = 0) # Expand to a 4th dimension as expected. We add a new dimension because it corresponds to a batch and the neural network expects a batch of inputs and so a new input also needs to be given as a batch.
result = classifier.predict(test_image)
training_set.class_indices # Tells the mapping, what is cat and what is dog(0 or 1).
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)