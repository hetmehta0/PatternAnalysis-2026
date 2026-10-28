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

