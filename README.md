# CO-LOTIE: A Pipeline for Context-Aware Object Localization Using Long Text and Image Embeddings

 Goh Yi Xian, Lim Jia Yu, Soo Jin Xue, Tee Yee Taung, Phuah En Yi, Lim Tik Hong<br>


## 💻 How to Install

```
conda create -n colotie python=3.9
conda activate colotie
pip install -r requirements-latest.txt
```

### Dataset
Our manually curated dataset is available at <a href='./data'>./data</a>

### Pre-trained Weights Preparation
Please download pre-trained weights of [BERT](https://huggingface.co/google-bert/bert-base-uncased), [ViT-B-16-in21k](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k), and [ViT-B-32-in21k](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k) to cache-dir: <a href='./cache'>./cache</a>.

```
$cache/
|–– vit_base_patch16_224.augreg_in21k/
|–– vit_base_patch32_224.augreg_in21k/
|–– bert-base-uncased/
```

You may use <code>git</code> to clone these model repositories as below:

```
#Create dir cache
cd cache
git clone https://huggingface.co/google-bert/bert-base-uncased
git clone https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
git clone https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k
```

Finally, download the pretrained weights of <a href='https://huggingface.co/weiwu-ww/LoTLIP-ViT-B-16-100M'>LOTLIP</a> and place it in the root directory as <code>model.pt</code>. The weights for YOLOv8 will be downloaded automatically during runtime.


### How to evaluate
```
python main.py --cache-dir ./cache \
    --output_dir ./outputs \
    --data ./data \
    --sim-thres 0.0 \
    --iou-thres 0.75 \
    --seed 42
```

## 🔷 Bibtex


```bibtex
@inproceedings{CO-LOTIE,
  title={CO-LOTIE: A Pipeline for Context-Aware Object Localization Using Long Text and Image Embeddings},
  author={Goh Yi Xian, Lim Jia Yu, Soo Jin Xue, Tee Yee Taung, Phuah En Yi, Lim Tik Hong},
  booktitle={N/A},
  year={2025}
}
```

## ❤️ Acknowledgements

Our code is built on top of [LOTLIP](https://github.com/wuw2019/LoTLIP). Thanks for their nice work!