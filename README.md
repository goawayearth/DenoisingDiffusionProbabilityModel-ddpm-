# DenoisingDiffusionProbabilityModel
This may be the simplest implement of DDPM. I trained with CIFAR-10 daaset. The links of pretrain weight, which trained on CIFAR-10 are in the Issue 2. <br>
<br>
If you really want to know more about the framwork of DDPM, I have listed some papers for reading by order in the closed Issue 1.
<br>
<br>
Lil' Log is also a very nice blog for understanding the details of DDPM, the reference is 
"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Diffusion%20models%20are%20inspired%20by,data%20samples%20from%20the%20noise."
<br>
<br>
**HOW TO RUN**
* 1.  You can run Main.py to train the UNet on CIFAR-10 dataset. After training, you can set the parameters in the model config to see the amazing process of DDPM.
* 2.  You can run MainCondition.py to train UNet on CIFAR-10. This is for DDPM + Classifier free guidence.

Some generated images are showed below:

* 1. DDPM without guidence:

![Generated Images without condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledNoGuidenceImgs.png)

* 2. DDPM + Classifier free guidence:

![Generated Images with condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledGuidenceImgs.png)

---

# 去噪扩散概率模型
这可能是 DDPM 最简单的实现。我使用 CIFAR-10 数据集进行了训练。预训练权重的链接（在 CIFAR-10 上训练）在问题 2 中。<br>
<br>
如果你真的想了解更多关于 DDPM 框架的信息，我在关闭的问题 1 中按顺序列出了一些论文供阅读。
<br>
<br>
Lil' Log 也是一个非常好的博客，可以帮助理解 DDPM 的细节，参考链接是 
"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Diffusion%20models%20are%20inspired%20by,data%20samples%20from%20the%20noise."
<br>
<br>
**如何运行**
* 1. 你可以运行 Main.py 来训练 CIFAR-10 数据集上的 UNet。训练后，你可以在模型配置中设置参数，以查看 DDPM 的惊人过程。
* 2. 你可以运行 MainCondition.py 来训练 CIFAR-10 上的 UNet。这是用于 DDPM + 无分类器引导。

下面显示了一些生成的图像：

* 1. 没有引导的 DDPM：

![没有条件生成的图像](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledNoGuidenceImgs.png)

* 2. DDPM + 无分类器引导：

![有条件生成的图像](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledGuidenceImgs.png)