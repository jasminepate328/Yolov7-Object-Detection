from pathlib import Path
import tensorflow as tf
import numpy as np

class CarsDatasetAdaptor(tf.data.Dataset):
    def __init__(
        self,
        images_dir_path,
        annotations_dataframe,
        transforms=None
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.transforms = transforms

        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_image_idx = {
            v: k for k, v in self.image_idx_to_image_id.items()
        }

    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)

    def __getitem__(self, index):
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        print('image_info', image_info)
        file_name = image_info.image.values[0]
        assert image_id == image_info.image_id.values[0]

        image = tf.keras.utils.load_img(
            self.images_dir_path / file_name, 
            color_mode='rgb'
        )
        # image_arr = tf.keras.utils.img_to_array(image)
        image = np.array(image)
        image_hw = image.shape[:2]

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            print('xyxy_bboxes', xyxy_bboxes)
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=xyxy_bboxes,
                labels=class_ids
            )
            image = transformed[image]
            xyxy_bboxes = np.array(transformed['bboxes'])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw

    def _inputs(self):
        return super()._inputs
    
    def element_spec(self):
        return super().element_spec
