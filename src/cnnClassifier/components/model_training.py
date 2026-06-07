import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common import save_json


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path, compile=False)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        training_data_dir = self.config.training_data
        if os.path.exists(os.path.join(training_data_dir, "kidney-ct-scan-image")):
            training_data_dir = os.path.join(training_data_dir, "kidney-ct-scan-image")

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=training_data_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = max(
            1, self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = max(
            1, self.valid_generator.samples // self.valid_generator.batch_size
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)

        # Save class indices
        class_indices = self.train_generator.class_indices
        class_indices_path = Path(self.config.trained_model_path).parent / "class_indices.json"
        save_json(path=class_indices_path, data=class_indices)

