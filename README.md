# FastDepth Monocular Depth Estimation
This repo provides an implementation of the monocular depth estimation architecture presented in the paper "FastDepth: Fast Monocular Depth Estimation on Embedded Systems" by Diana Wofk et al.
The architecture was implemented in PyTorch. Training was performed on the NYU DepthV2 dataset. Trained model file, as well as a small compiled web demo is also provided.

![Example of model predictions](illustrations/fast_depth_illustration.png)

## Implementation Details
### Data Preprocessing
For the RGB input images, the Eigen crop is applied to account for the known dataset artefacts. Additionally, color jitter and random horizontal flips are done.
Nearest Neighbour Interpolation is used for the ground truth rescaling to preserve the sharp depth drops and avoid blurred transitions.
### Architecture
Architecture was done according to the paper, with the best performing encoder and decoder chosen for implementation.
Pretrained MobileNetV2 was used as an encoder, with weight freezing to prevent fitting to a degenerate decoder during initial epochs.
