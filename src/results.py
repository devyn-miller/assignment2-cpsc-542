import tensorflow as tf
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt


label = [0, 0]
prediction = [[-3., 0], [-3, 0]]
sample_weight = [1, 10]

# Convert lists to tensors
label_tensor = tf.constant(label)
prediction_tensor = tf.constant(prediction)
sample_weight_tensor = tf.constant(sample_weight)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)
# Use tensors instead of lists
loss_value = loss(label_tensor, prediction_tensor, sample_weight_tensor).numpy()
print(loss_value)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Generate predictions
    y_true = np.array([])
    y_pred = np.array([])
    for datapoint in test_dataset.unbatch().batch(1).take(-1):
        image, mask = datapoint
        pred_mask = model.predict(image)
        y_true = np.append(y_true, mask.numpy().flatten())
        y_pred = np.append(y_pred, np.argmax(pred_mask, axis=-1).flatten())

    # Calculate metrics
    conf_mat = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    iou = tf.keras.metrics.MeanIoU(num_classes=3)
    iou.update_state(y_true, y_pred)
    iou_score = iou.result().numpy()

    # Print metrics
    print(f'Confusion Matrix:\n{conf_mat}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'IoU: {iou_score}')

    # Plot metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.bar(['Precision', 'Recall', 'F1', 'IoU'], [precision, recall, f1, iou_score])
    plt.title('Metrics')
    plt.show()


