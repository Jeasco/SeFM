# Separating Anything from Image in Context (in submission)

 :heart_eyes: :heart_eyes: SeFM is a novel generalist foundation model for separating anything from image with only a handy demonstration example, which exhibits groundbreaking generalization capability to open-world unseen visual components. :heart_eyes: :heart_eyes: 
<table>
  <tr>
    <td> <img src = "figures/inference.png"> </td>
  </tr>
</table>


> **Abstract:***Separating undesired visual components (e.g. shadow and watermark) from images has long been a hot spot in the computer vision community. However, existing decomposition methods only perform well in separating in-domain known components while failing to generalize to unseen out-of-domain components. How to unifiedly separate open-world arbitrary visual components from images has remained unprobed. In this paper, we propose SeFM, a novel generalist foundation model for separating anything from image in context, which exhibits groundbreaking generalization capability to open-world unseen visual components. In particular, we advocate demarcating different visual components with semantics and tame a tailored in-context learning paradigm to cultivate the model to explicitly refer to the semantics of the separated visual components in the demonstration example to separate semantically identical visual components in the query image. Noteworthy, this learning paradigm enables the model master class-agnostic semantic matching capacity, i.e., discerning which visual components in the context are semantically identical to be separated, irrespective of the specific class of what those semantics represent. Thus, during inference, we can stitch a handy demonstration example as reference wherein visual components of the desired semantic are removed, SeFM can automatically separate semantically identical components in the query image even if the semantic is out-of-domain. Without bells and whistles, our SeFM can separate open-world arbitrary visual components in context and yield competitive performance compared to state-of-the-art task-specific models, on ten representative vision tasks ranging from low-level image decomposition to high-level potential applications. Furthermore, evaluations on few-shot learning and transfer learning spotlight unparalleled performance gains with pre-trained SeFM across diverse separation-related downstream tasks, signifying substantial potential in advancing foundation models within low-level visions.* 

## Framework Architecture
<table>
  <tr>
    <td> <img src = "figures/SeFM.png"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of SeFM</b></p></td>
  </tr>
</table>


## `Installation`
`The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).`

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python timm einops ptflops PIL argparse
```

## Training

```
bash sefm_training.sh
```


## Evaluation

1. Download the pre-trained model and place it in `./checkpoints/`

2. Place the test image in `./test/`

3. Run
```
python in_context_inference_demo.py
```
4. Visual results wii be saved in results



