import pandas as pd
from pathlib import Path
from train_cars import load_cars_df
from cars_dataset_adaptor import CarsDatasetAdaptor
from utils.plotting import show_image

data_path = Path("data/cars")
images_path = data_path / "training_images"
annotations_file_path = data_path / "train_solution_bounding_boxes.csv"

train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)

print(f"# of annotated images in the training set: {len(train_df.query('has_annotation == True').image.unique())}")
print(f"# Background images in training set: {len(train_df.query('has_annotation == False').image.unique())}")
print(f"# of images in the training set, {len(train_df.image.unique())}")

print(f"# of annotated images in the validation set: {len(valid_df.query('has_annotation == True').image.unique())}")
print(f"# Background images in validation set: {len(valid_df.query('has_annotation == False').image.unique())}")
print(f"# of images in the validation set: {len(valid_df.image.unique())}")

ds = CarsDatasetAdaptor(images_path, train_df)
idx = 4
image, xyxy_bboxes, class_ids, image_idx, image_size = ds[idx]
show_image(image, xyxy_bboxes.tolist(), [lookups['class_id_to_label'][int(c)] for c in class_ids])
