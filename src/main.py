from data_utils.dataset import Dataset

dataset_directory = r"C:\Users\Lukas\source\studium\sample"
file_type = "obj"
points = 2048
# dataset_type = "test"


dataset = Dataset(dataset_directory, file_type, points, "train")
augmentations = ["RandomNoise"]
train_ds, train_annotations = dataset.create_dataset(augmentations)


dataset = Dataset(dataset_directory, file_type, points, "test")
augmentations = []
test_ds, test_annotations = dataset.create_dataset(augmentations)
print(test_annotations)
print(test_ds)
print(train_annotations)
print(train_ds)
