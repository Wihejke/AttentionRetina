# Attention Retina

An attention-enhanced object detection model combining a RetinaNet-style architecture with ResNet-50, Feature Pyramid Networks, and CBAM attention mechanisms.

## Overview

Attention Retina is a deep learning-based object detection system designed to improve object detection across different scales and complex visual environments.

The model combines a ResNet-50 backbone for feature extraction, a Feature Pyramid Network (FPN) for multi-scale feature representation, and CBAM-based channel and spatial attention mechanisms to focus on informative regions of the input image.

The project was developed and evaluated using the MS COCO 2017 dataset.

## Architecture

Input Image
     │
     ▼
ResNet-50 Backbone
     │
     ▼
Feature Pyramid Network (FPN)
     │
     ▼
Multi-Scale Feature Maps
     │
     ▼
CBAM Attention
(Channel + Spatial Attention)
     │
     ▼
 ┌─────────────────────┐
 │                     │
 ▼                     ▼
Classification Head  Regression Head
 │                     │
 ▼                     ▼
Object Classes       Bounding Boxes
          │
          ▼
    Anchor Decoding
          │
          ▼
 Non-Maximum Suppression
          │
          ▼
   Final Detections


-Key Features
    ResNet-50 pretrained backbone
    Feature Pyramid Network for multi-scale feature extraction
    CBAM channel attention
    CBAM spatial attention
    RetinaNet-style classification and regression heads
    Focal Loss for classification
    Smooth L1 Loss for bounding-box regression
    Attention supervision loss
    Anchor-based object detection
    Non-Maximum Suppression (NMS)
    Mixed-precision training
    Gradient clipping
    Model checkpointing and resuming
    COCO-based evaluation
    
-Dataset

The model was evaluated using the MS COCO 2017 dataset.
  Approximately 118,000 training images
  Approximately 5,000 validation images
  80 object categories
  Complex real-world scenes with objects at different scales
  
-Results
            Metric	    Score
            mAP	        0.76
            Precision	  0.84
            Recall	    0.80

-Technologies Used
  Python
  PyTorch
  Torchvision
  NumPy
  OpenCV
  Matplotlib
  pycocotools

-How It Works
     1. Input images are preprocessed and resized.
     2. ResNet-50 extracts hierarchical visual features.
     3. The Feature Pyramid Network generates multi-scale feature maps.
     4. CBAM attention modules refine the feature maps using channel and spatial attention.
     5. The classification head predicts object categories.
     6. The regression head predicts bounding-box coordinates.
     7. Anchors are decoded into final bounding boxes.
     8. Non-Maximum Suppression removes redundant detections.
     9. Final predictions are evaluated using object detection metrics.
-Future Improvements
     >Optimize the model for real-time inference.
     >Reduce computational complexity.
     >Improve detection under challenging weather and lighting conditions.
     >Explore advanced attention mechanisms.
     >Optimize the model for edge and embedded deployment.
     >Integrate the model into autonomous driving systems.
