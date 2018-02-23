# copying images to training, validation, and test directories.
import os, shutil
from utils import mkdir_pwd, copy_pwd

# path to the directory where the original dataset was uncompressed.
original_data_cats_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/PetImages/Cat'
original_data_dogs_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/PetImages/Dog'

# directory where you'll store your smaller dataset
base_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data'
os.mkdir(base_dir)

# directories for the training, validation, and test splits
train_dir = mkdir_pwd(base_dir, 'train')
validation_dir = mkdir_pwd(base_dir, 'validation')
test_dir = mkdir_pwd(base_dir, 'test')

# directory with training cat pictures
train_cats_dir = mkdir_pwd(train_dir, 'cats')

# directory with training dog pictures
train_dogs_dir = mkdir_pwd(train_dir, 'dogs')

# directory with validation cat pictures
validation_cats_dir = mkdir_pwd(validation_dir, 'cats')

# directory with validation dog pictures
validation_dogs_dir = mkdir_pwd(validation_dir, 'dogs')

# directory with test cat pictures
test_cats_dir = mkdir_pwd(test_dir, 'cats')

# directory with test dog pictures
test_dogs_dir = mkdir_pwd(test_dir, 'dogs')

# copies the first 1000 cat images to train_cats_dir
copy_pwd('{}.jpg', 0, 1000, original_data_cats_dir, train_cats_dir)

# copies the next 500 cat images to validation_cats_dir
copy_pwd('{}.jpg', 1000, 1500, original_data_cats_dir, validation_cats_dir)

# copies the next 500 cat images to test_cats_dir
copy_pwd('{}.jpg', 1500, 2000, original_data_cats_dir, test_cats_dir)

# copies the first 1000 dog images to train_dogs_dir
copy_pwd('{}.jpg', 0, 1000, original_data_dogs_dir, train_dogs_dir)

# copies the next 500 dog images to validation_dogs_dir
copy_pwd('{}.jpg', 1000, 1500, original_data_dogs_dir, validation_dogs_dir)

# copies the next 500 dog images to test_dogs_dir
copy_pwd('{}.jpg', 1500, 2000, original_data_dogs_dir, test_dogs_dir)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

