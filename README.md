
eisai-anime-interpolator
========================

**Improving the Perceptual Quality of 2D Animation Interpolation**  
Shuhong Chen[\*](https://shuhongchen.github.io/), Matthias Zwicker[\*](https://www.cs.umd.edu/~zwicker/)  
ECCV2022  
\[[arxiv](https://arxiv.org/abs/2111.12792)\]
\[[github](https://github.com/ShuhongChen/eisai-anime-interpolator)\]  

*Traditional 2D animation is labor-intensive, often requiring animators to manually draw twelve illustrations per second of movement.  While automatic frame interpolation may ease this burden, 2D animation poses additional difficulties compared to photorealistic video.  In this work, we address challenges unexplored in previous animation interpolation systems, with a focus on improving perceptual quality.  Firstly, we propose SoftsplatLite (SSL), a forward-warping interpolation architecture with fewer trainable parameters and better perceptual performance.  Secondly, we design a Distance Transform Module (DTM) that leverages line proximity cues to correct aberrations in difficult solid-color regions.  Thirdly, we define a Restricted Relative Linear Discrepancy metric (RRLD) to automate the previously manual training data collection process.  Lastly, we explore evaluation of 2D animation generation through a user study, and establish that the LPIPS perceptual metric and chamfer line distance (CD) are more appropriate measures of quality than PSNR and SSIM used in prior art.*


