HipMRI Anatomical Segmentation Using 2D and 3D U-Net Architectures

1) Introduction

1.1
 The purpose of medical segmentation using computer vision is to easily idenitfy areas of need to detect lesions, their size, location and relationship with nearby areas.  

 MRI is used because it excels at differentiating subtle variations in dense soft tissues—such as distinguishing gray matter, white matter, and cerebrospinal fluid in the brain, or mapping muscles, ligaments, and cartilage in joints. Moreover It provides an accurate 3D cross-sectional volumes across any spatial plane making. 

Even tho MRI is capable of providing multiplanar anatomical information, there can still be areas that hard to differentiate with some planes having lower resolution than others 
- 2D limitations 
The limitations with 2D implentation is that each slice is processed independently. Additionally it can see the cross-sectional area but not its extent into neighbouring slices. Moreover it cannot directly determine whether a small region is the beginning of an organ or its end or a complete different one with a siilar apperance. It might also produce inconsistent boundaries between adjacent slices
- 3D oppuritnity 
The benefit of using a 3D U-net is that it can process a block of neighbouring slices simultaneously, as well as learn how the organ's shape changes along the depth axis. Moreover it learns continuity, cruvature and spatial relationship across slices. It may produce more consistent anatomical segmentation. 

1.2 
RQ: Does incorporating 3D spatial context improve multi-class MRI segmentation when compared to a standard 2D U-Net segmentation baseline and does and improved 3D U-Net using residual connections provide enough improvement for the increased computational cost?

1.3 
Hypothesis: 
IF A 3D U-Net uses spatial information from neighbouring MRI slices THEN It may produce more accurate segmentation than a standard 2D U-Net BECAUSE Organs are three-dimensional structures whose cross-sectional shapes and boundaries change across depth. Learning these relationships may help identify anatomical boundaries that are ambiguous in individual 2D slices.

1.4 Eval approach 
Performance is evaluated using mean foreground Dice and per-class Dice scores on the validation dataset. Computational costs, including training time and model complexity, are also considered to determine whether any improvements in segmentation accuracy justify the additional computational resources required.

2) Dataset and processing 
2.1 DataSet Overview
The HipMRI dataset contains 211 MRI Volumes from 38 patients. Each Volume has a corresponding segementation and the data is stored in NIfTI (.nii.gz) format. Volume's dimensions are: 256x256x128 or 256x256x144. There are six segmentation labels: background, bladder, body, bone, rectum and prostate. Generaly the bladder and body occupy are relatively large regions whereas the rectum and prostate are smaller regions. Furthermore each structure has different geometry and making segmentation difficulty vary beteen classes. Finally these differences allows us to investigate whether 3D spatial context particulary benefits smaller or more complex anatomical structures 

2.2 Dataset Splitting

| Split | Patients | MRI Volumes |
|---|---:|---:|
| Training | 26 | 136 |
| Validation | 6 | 34 |
| Testing | 6 | 41 |
| **Total** | **38** | **211** |

**Key points:**

- Patient-level splitting prevents scans belonging to the same patient from appearing across different subsets, reducing the risk of data leakage.
- The **training set** is used to optimise the model's parameters.
- The **validation set** is used to evaluate performance during training and select the best-performing model.
- The **test set** remains held out for final evaluation, providing an independent assessment of model performance.

2.3 Preprocessing
2D U-Net:
- MRI volumes divided into individual 2D axial slices.
- Images paired with corresponding segmentation masks.
- Individual slices fed into the model.
- Predictions reassembled into volumes for validation.
3D U-Net:
- MRI volumes processed as volumetric data.
- Extract 96 × 96 × 96 voxel patches.
- Use class-balanced sampling with foreground probability 0.33.
- Patch sampling helps expose the network to less frequently represented structures.
- Use overlapping sliding-window inference to evaluate full volumes.

(run command in gpu lab to get table and paste here)


3.0 
User need and scope. An imaging specialist needs a prototype that drafts prostate contours for review. Compare normal-difficulty 2D Improved U-Net with standard 2D U-Net on the same held-out HipMRI patients. Investigate whether residual multi-scale features reduce difficult boundary errors, despite the absence of 3D continuity. This prototype is not approved for clinical use.
Testable acceptance criteria (proposed). (1) Zero patient overlap and no reused files across partitions; reject misaligned pairs and unknown labels. (2) Held-out patient-macro prostate Dice >= 0.75, with pooled and per-class Dice/IoU also reported. (3) Review 3-5 real boundary failures, including apex/base and adjacent-organ leakage, using input/truth/prediction/FP-FN panels. (4) On the allocated GPU, target peak allocated memory <= 8 GiB and warmed forward latency <= 1 second per 256x256 slice at batch 1. These are proposed resource limits, not measured Rangpur results. (5) Compare both models using the identical split, preprocessing, seed and training budget.
Model choice and course concepts. Use 2D Nifti slices, a verified fixed multi-class vocabulary, per-slice normalization and patient-level partitions. Convolutions, pooling/striding, skip connections, residual learning, dropout and gradient supervision connect course concepts to the architecture. 
Preliminary evidence. The downloaded data contains 12,660 aligned slice pairs from 38 patients. The supplied train/validation/test partitions contain 31/4/3 patients and 11,460/660/540 slices, with no patient overlap. Every pair passed shape, affine and finite-value checks. Labels 0-5 match the published order: background, bladder, body, bone, rectum and prostate. Prostate occupies 1,766,549 of 415,088,640 pixels (0.426%), confirming severe imbalance. Most slices are 256x128; 60 training slices are 256x144. Visual checks of positive examples from every split showed the label-5 contour aligned with the prostate. Both models also completed a one-epoch CPU pipeline check on 114 real slices, including checkpoint reload, native-grid evaluation and prediction export.
Risks, budget and fallback. Main risks are small-cohort variance, resize-related boundary loss, extreme class imbalance and 2D continuity failures. The next experiment is a short representative optimization run on the full training partition, followed by one 50-epoch run per model at width 16, 256x256, batch 8; measure a real epoch before reserving GPU hours. Use the same reduced batch size for both if memory is insufficient. Retain the baseline if the improved model lacks meaningful boundary benefit; discuss any difficulty change with the tutor. No superiority or deployment recommendation is justified yet. Present this review at the compulsory practical consultation.

Following the feasibility review consultation, the tutor suggested investigating the 3D aspect of MRI segmentation, which I had not initially considered. My original plan was to compare a standard 2D U-Net with an improved 2D U-Net using residual connections. Based on this feedback, I revised the project to use the 2D U-Net as a baseline, followed by a standard 3D U-Net and an improved 3D U-Net with residual connections. This revised approach allows me to investigate whether incorporating spatial information across neighbouring MRI slices improves segmentation performance, while also evaluating the effect of residual connections.

## 4. Methodology

Three U-Net-based architectures were used to investigate the effect of spatial context and architectural improvements on multi-class MRI segmentation: a standard 2D U-Net, a standard 3D U-Net, and an improved 3D U-Net incorporating residual connections. The 2D model serves as the baseline, while the 3D models investigate whether learning spatial relationships between neighbouring MRI slices improves segmentation performance.

4.1 Standard 2D U-Net

The initial baseline uses a standard 2D U-Net architecture consisting of an encoder, bottleneck, decoder, and skip connections. The encoder progressively extracts higher-level features through convolutional blocks while reducing the spatial resolution of the input image. Each convolutional block contains two convolution operations with batch normalisation and ReLU activation. Max pooling is used for downsampling. The bottleneck processes the encoded features at the lowest spatial resolution, allowing the network to learn more complex representations of the input. The decoder progressively restores the original spatial resolution using transposed convolutions. Skip connections concatenate features from the corresponding encoder stages with decoder features, helping recover spatial details lost during downsampling. This is particularly important for segmentation, where accurately identifying anatomical boundaries requires both high-level context and local information.

Two configurations were implemented:

| Model | Encoder Channels | Bottleneck Channels | Trainable Parameters |
|---|---|---:|---:|
| 2D U-Net 256 | 32, 64, 128, 256 | 256 | 5,140,166 |
| 2D U-Net 512 | 32, 64, 128, 256 | 512 | 7,762,630 |

Both configurations process individual axial MRI slices as single-channel images and produce six output channels corresponding to the segmentation classes. Although this approach can learn spatial relationships within each slice, it cannot directly use anatomical information from neighbouring slices. Consequently, structures that change in shape or size across the MRI volume may be more difficult to identify using individual cross-sections alone.

4.2 Standard 3D U-Net

To investigate whether volumetric information improves segmentation, the 2D architecture was extended into a standard 3D U-Net.

The 3D model follows the same encoder-decoder structure, but replaces two-dimensional convolutions, pooling, and transposed convolutions with their three-dimensional equivalents. This allows the network to extract features across the depth, height, and width of the input volume rather than processing each MRI slice independently.

The 3D model therefore has access to neighbouring slices within each input patch and can learn how anatomical structures change across depth. This may be beneficial for identifying curved anatomical boundaries and maintaining spatial continuity between consecutive slices.

Two bottleneck configurations were implemented:

| Model | Encoder Channels | Bottleneck Channels | Trainable Parameters |
|---|---|---:|---:|
| 3D U-Net 256 | 32, 64, 128, 256 | 256 | 14,975,238 |
| 3D U-Net 512 | 32, 64, 128, 256 | 512 | 22,578,438 |

3D Patch Sampling

Instead of processing entire MRI volumes during training, the model uses patches of 96 × 96 × 96 voxels.

Patch-based training reduces the memory requirements associated with full-volume 3D convolution and allows multiple training samples to be extracted from each MRI scan.

Class-balanced patch sampling was used with a foreground sampling probability of 0.33 to increase the representation of anatomical structures that may otherwise appear infrequently in uniformly sampled patches.

A preliminary sampling analysis compared 64³ and 96³ patches. The 96³ class-balanced configuration was selected because it provided greater spatial context and more frequent inclusion of smaller anatomical classes, particularly the prostate and rectum.

Full-Volume Validation

During validation, predictions are generated for entire MRI volumes using sliding-window inference.

The volume is divided into overlapping 3D patches, each of which is processed by the model. Predictions from overlapping regions are combined using uniform averaging to reconstruct the complete volumetric segmentation.

An overlap of 50% was used to provide coverage across the volume and reduce the dependence of predictions on individual patch boundaries.

This allows the model to be trained using manageable patches while still being evaluated on complete MRI volumes.

4.3 Improved Residual 3D U-Net

An improved 3D U-Net architecture was developed to investigate whether residual connections can further improve segmentation performance.

The standard U-Net uses skip connections between encoder and decoder stages to preserve spatial information. Residual connections operate differently: they introduce shortcut paths within convolutional blocks, allowing the block's input features to be added to its transformed output.

The residual operation can be expressed as:

\[
y = \operatorname{ReLU}(F(x) + S(x))
\]

where \(F(x)\) represents the convolutional transformation and \(S(x)\) represents the identity or projection shortcut used to match feature dimensions.

These connections provide an additional pathway for information and gradients to propagate through the network, potentially making optimisation more effective.

The improved model retains the overall 3D U-Net encoder-decoder architecture, volumetric input format, six output classes, and long skip connections.
