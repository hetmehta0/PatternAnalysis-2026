HipMRI Anatomical Segmentation Using 2D and 3D U-Net Architectures

1) Introduction
1.1
 - What/why do we need medical segmentation. What purpose does it solve 

 - Why MRI (ideally most complete type of scan with most accurate data etc blah blah )
  
-  Introduce challange (hard to idenitfy boundary for different body parts in scans)

- 2D limitations 

- 3D oppuritnity 

1.2 
RQ: Does incorporating 3D spatial context improve multi-class MRI segmentation when compared to a standard 2D U-Net segmentation baseline and does and improved 3D U-Net using residual connections provide enough improvement for the increased computational cost?

1.3 
Hypothesis: 
IF A 3D U-Net uses spatial information from neighbouring MRI slices THEN It may produce more accurate segmentation than a standard 2D U-Net BECAUSE Organs are three-dimensional structures whose cross-sectional shapes and boundaries change across depth. Learning these relationships may help identify anatomical boundaries that are ambiguous in individual 2D slices.

1.4 Eval approach 
- Compare 2D, normal 3d and improved 3d 
- measure validation mean foreground dice and per class dice and computation cost (does addtional computation time/cost provide a enough improvement)

