import keras
from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import numpy as np

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)

training_filenames = []
GT_training = []
batch_size = 1

validation_filenames = []
GT_validation = []

my_training_batch_generator = My_Generator(training_filenames, GT_training, batch_size)
my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)

model.fit_generator(generator=my_training_batch_generator,
                                        steps_per_epoch=(num_training_samples // batch_size),
                                        epochs=num_epochs,
                                        verbose=1,
                                        validation_data=my_validation_batch_generator,
                                        validation_steps=(num_validation_samples // batch_size),
                                        use_multiprocessing=True,
                                        workers=16,
                                        max_queue_size=32)



