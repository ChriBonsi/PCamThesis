import h5py


def count_pictures_in_h5(file_path):
    # Open the .h5 file in read mode
    with h5py.File(file_path, 'r') as h5_file:
        # Inspect the structure of the file
        keys = list(h5_file.keys())
        print(f"Keys in the file: {keys}")

        # Assuming the images are stored in a dataset, let's inspect one of the keys
        for key in keys:
            data = h5_file[key]
            print(f"{key}: {data.shape}")

        # If the images are stored in a single dataset, get its shape
        # Replace 'dataset_name' with the actual name if known
        dataset_name = keys[0]  # Assuming images are stored under the first key
        images_dataset = h5_file[dataset_name]
        number_of_images = images_dataset.shape[0]

        return number_of_images


# Path to your .h5 file
train_path = 'data/PCam/train/pcam/camelyonpatch_level_2_split_train_x.h5'
number_of_train_pictures = count_pictures_in_h5(train_path)
print(f"Number of pictures in the training set: {number_of_train_pictures}")

print("\n-----------------------\n")

# Path to your .h5 file
test_path = 'data/PCam/test/pcam/camelyonpatch_level_2_split_test_x.h5'
number_of_test_pictures = count_pictures_in_h5(test_path)
print(f"Number of pictures in the testing set: {number_of_test_pictures}")
