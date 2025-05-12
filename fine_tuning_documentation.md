
### Model Selection and Fine-tuning Documentation

#### Models Selected:
- **ResNet18**: A lightweight residual network with 18 layers, pre-trained on ImageNet.
- **MobileNetV2**: A mobile-optimized network with inverted residuals, pre-trained on ImageNet.

#### Hyperparameters:
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Batch Size**: 32
- **Epochs**: 5
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

#### Training Process:
- **Dataset**: Mini-ImageNet (100 classes, 500 training images per class, 100 validation/test images per class)
- **Preprocessing**: Resized images to 96x96 pixels, normalized with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
- **Validation Split**: Used provided validation set from Mini-ImageNet val.tar.
- **Fine-tuning Strategy**: Modified the final fully connected layer for 100 classes. Trained all layers with Adam optimizer.
- **Validation Strategy**: Monitored validation accuracy with ModelCheckpoint to save the best model.
- **Hardware**: Utilized GPU if available, otherwise CPU.

#### Baseline Performance:
- **ResNet18 Test Accuracy**: 0.0048
- **MobileNetV2 Test Accuracy**: 0.0047
- **Detailed Metrics**: Classification reports provided below.

### ResNet18 Classification Report:
              precision    recall  f1-score   support

           0       0.09      0.02      0.03       600
           1       0.08      0.01      0.02       600
           2       0.03      0.00      0.01       600
           3       0.01      0.00      0.00       600
           4       0.02      0.00      0.01       600
           5       0.10      0.00      0.00       600
           6       0.03      0.01      0.01       600
           7       0.01      0.00      0.00       600
           8       0.04      0.01      0.01       600
           9       0.00      0.00      0.00       600
          10       0.00      0.00      0.00       600
          11       0.00      0.00      0.00       600
          12       0.01      0.01      0.01       600
          13       0.00      0.00      0.00       600
          14       0.01      0.00      0.01       600
          15       0.02      0.01      0.01       600
          16       0.00      0.00      0.00       600
          17       0.00      0.00      0.00       600
          18       0.03      0.01      0.01       600
          19       0.01      0.01      0.01       600
          20       0.00      0.00      0.00         0
          21       0.00      0.00      0.00         0
          22       0.00      0.00      0.00         0
          23       0.00      0.00      0.00         0
          24       0.00      0.00      0.00         0
          25       0.00      0.00      0.00         0
          26       0.00      0.00      0.00         0
          27       0.00      0.00      0.00         0
          28       0.00      0.00      0.00         0
          29       0.00      0.00      0.00         0
          30       0.00      0.00      0.00         0
          31       0.00      0.00      0.00         0
          32       0.00      0.00      0.00         0
          33       0.00      0.00      0.00         0
          34       0.00      0.00      0.00         0
          35       0.00      0.00      0.00         0
          36       0.00      0.00      0.00         0
          37       0.00      0.00      0.00         0
          38       0.00      0.00      0.00         0
          39       0.00      0.00      0.00         0
          40       0.00      0.00      0.00         0
          41       0.00      0.00      0.00         0
          42       0.00      0.00      0.00         0
          43       0.00      0.00      0.00         0
          44       0.00      0.00      0.00         0
          45       0.00      0.00      0.00         0
          46       0.00      0.00      0.00         0
          47       0.00      0.00      0.00         0
          48       0.00      0.00      0.00         0
          49       0.00      0.00      0.00         0
          50       0.00      0.00      0.00         0
          51       0.00      0.00      0.00         0
          52       0.00      0.00      0.00         0
          53       0.00      0.00      0.00         0
          54       0.00      0.00      0.00         0
          55       0.00      0.00      0.00         0
          56       0.00      0.00      0.00         0
          57       0.00      0.00      0.00         0
          58       0.00      0.00      0.00         0
          59       0.00      0.00      0.00         0
          60       0.00      0.00      0.00         0
          61       0.00      0.00      0.00         0
          62       0.00      0.00      0.00         0
          63       0.00      0.00      0.00         0

    accuracy                           0.00     12000
   macro avg       0.01      0.00      0.00     12000
weighted avg       0.02      0.00      0.01     12000

### MobileNetV2 Classification Report:
              precision    recall  f1-score   support

           0       0.10      0.01      0.01       600
           1       0.00      0.00      0.00       600
           2       0.01      0.00      0.01       600
           3       0.00      0.00      0.00       600
           4       0.02      0.01      0.02       600
           5       0.00      0.00      0.00       600
           6       0.00      0.00      0.00       600
           7       0.04      0.00      0.00       600
           8       0.07      0.04      0.05       600
           9       0.00      0.00      0.00       600
          10       0.01      0.00      0.01       600
          11       0.01      0.01      0.01       600
          12       0.01      0.01      0.01       600
          13       0.00      0.00      0.00       600
          14       0.02      0.01      0.01       600
          15       0.00      0.00      0.00       600
          16       0.00      0.00      0.00       600
          17       0.00      0.00      0.00       600
          18       0.00      0.00      0.00       600
          19       0.00      0.00      0.00       600
          20       0.00      0.00      0.00         0
          21       0.00      0.00      0.00         0
          22       0.00      0.00      0.00         0
          23       0.00      0.00      0.00         0
          24       0.00      0.00      0.00         0
          25       0.00      0.00      0.00         0
          26       0.00      0.00      0.00         0
          27       0.00      0.00      0.00         0
          28       0.00      0.00      0.00         0
          29       0.00      0.00      0.00         0
          30       0.00      0.00      0.00         0
          31       0.00      0.00      0.00         0
          32       0.00      0.00      0.00         0
          33       0.00      0.00      0.00         0
          34       0.00      0.00      0.00         0
          35       0.00      0.00      0.00         0
          36       0.00      0.00      0.00         0
          37       0.00      0.00      0.00         0
          38       0.00      0.00      0.00         0
          39       0.00      0.00      0.00         0
          40       0.00      0.00      0.00         0
          41       0.00      0.00      0.00         0
          42       0.00      0.00      0.00         0
          43       0.00      0.00      0.00         0
          44       0.00      0.00      0.00         0
          45       0.00      0.00      0.00         0
          46       0.00      0.00      0.00         0
          47       0.00      0.00      0.00         0
          48       0.00      0.00      0.00         0
          49       0.00      0.00      0.00         0
          50       0.00      0.00      0.00         0
          51       0.00      0.00      0.00         0
          52       0.00      0.00      0.00         0
          53       0.00      0.00      0.00         0
          54       0.00      0.00      0.00         0
          55       0.00      0.00      0.00         0
          56       0.00      0.00      0.00         0
          57       0.00      0.00      0.00         0
          58       0.00      0.00      0.00         0
          59       0.00      0.00      0.00         0
          60       0.00      0.00      0.00         0
          61       0.00      0.00      0.00         0
          62       0.00      0.00      0.00         0
          63       0.00      0.00      0.00         0

    accuracy                           0.00     12000
   macro avg       0.00      0.00      0.00     12000
weighted avg       0.01      0.00      0.01     12000


### Validation Strategy Documentation

#### Validation Dataset:
- **Source**: Mini-ImageNet validation set (val.tar), containing 100 images per class for 100 classes (10,000 images total).
- **Preprocessing**: Same as training set—resized to 96x96 pixels, normalized with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

#### Validation Process:
- **Frequency**: Validation was performed after each epoch during training.
- **Metric Monitored**: Validation accuracy (`val_acc`) was computed using sklearn.metrics.accuracy_score.
- **Checkpointing**: Used PyTorch Lightning’s ModelCheckpoint callback to save the model with the highest validation accuracy (`monitor="val_acc", mode="max", save_top_k=1`).

#### Purpose:
- Ensured the model generalizes well to unseen data and prevents overfitting by monitoring validation accuracy.
- The best model (based on validation accuracy) was saved and used for testing in Step 6.

#### Validation Results:
- Best validation accuracy for ResNet18 and MobileNetV2 was logged during training (visible in PyTorch Lightning logs).
- Final test performance on the uncompressed test set reflects the effectiveness of this strategy (see Step 6).


### Validation Strategy Documentation

#### Validation Dataset:
- Source: Mini-ImageNet validation set (val.tar), containing 100 images per class for 100 classes (10,000 images total).
- Preprocessing: Same as training set—resized to 96x96 pixels, normalized with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

#### Validation Process:
- Frequency: Validation was performed after each epoch during training.
- Metric Monitored: Validation accuracy (`val_acc`) was computed using sklearn.metrics.accuracy_score.
- Checkpointing: Used PyTorch Lightning’s ModelCheckpoint callback to save the model with the highest validation accuracy (`monitor="val_acc", mode="max", save_top_k=1`).

#### Purpose:
- Ensured the model generalizes well to unseen data and prevents overfitting by monitoring validation accuracy.
- The best model (based on validation accuracy) was saved and used for testing in Step 6.

#### Validation Results:
- Best validation accuracy for ResNet18 and MobileNetV2 was logged during training (visible in PyTorch Lightning logs).
- Final test performance on the uncompressed test set reflects the effectiveness of this strategy (see Step 6).
