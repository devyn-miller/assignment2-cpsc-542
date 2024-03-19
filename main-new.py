import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Importing modules from other files
from src.preprocessing import dataset, info, load_image
from src.model import unet_model
from src.train_predict import model, model_history, show_predictions, create_mask, DisplayCallback, EPOCHS, VAL_SUBSPLITS, VALIDATION_STEPS,OUTPUT_CLASSES, train_batches, test_batches, STEPS_PER_EPOCH
from src.results import evaluate_model
if __name__ == "__main__":
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    train_dataset = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    test_dataset = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test_dataset,
                        validation_steps=VALIDATION_STEPS)

    evaluate_model(model, test_dataset)
    # Generate Grad-CAM visualizations for a sample image
    for image, mask in test_dataset.take(1):
        # generate_gradcam(model, image)
        pass
        # plot accuracy for train and test by epoch
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # plot loss function for train and test by epoch
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()



