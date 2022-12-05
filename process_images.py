import pandas as pd
from pathlib import Path
from train_cars import load_cars_df


data_path = Path("data/cars")
images_path = data_path / "training_images"
annotations_file_path = data_path / "train_solution_bounding_boxes.csv"

train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)
