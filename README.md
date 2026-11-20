# Mooney-Minded

Project Part 1 of *Computational Visual Perception* (FAU, WS 2026/27): how well do pretrained vision systems understand Mooney images?

**Pipeline:** 100 COCO photos (10 Mooney-friendly classes x 10) -> Mooney images (Gaussian blur + fixed % white pixels) -> two pretrained systems, run on originals and Mooneys:

| | System | Question | Scored by |
|---|---|---|---|
| Result1 | **Qwen3-VL-4B-Instruct** (VLM) | Can it *name* what it sees? Can it be talked into a wrong answer? | open naming, 10-way choice, yes/no to true vs. wrong claim |
| Result2 | **SAM 3** (concept segmentation) | Told "giraffe", can it *find and outline* the giraffe? Does it hallucinate a "toilet"? | detection, IoU vs. COCO ground-truth mask, false detections |

Add-ons (slides only): polarity-inverted and upside-down Mooneys through the same probes.

## 0. Tool environment (once)

```bash
python3 -m venv .venv-tool && source .venv-tool/bin/activate
pip install -r requirements.txt
```

## 1. Dataset (~250 MB annotations + a few hundred photos)

```bash
python scripts/prepare_dataset.py
# -> data/Original/001_giraffe.png ... 100_toilet.png, data/masks/, data/labels.csv, data/selection_sheet.png
```

Look at `data/selection_sheet.png` (each tile shows its COCO id). To drop an image, put its id in `data/exclude.txt` and re-run.

## 2. Mooneys

```bash
python -m mooney_tool.gui     # Folder preview tab: pick ONE blur + white % by eye
python -m mooney_tool.cli -i data/Original -o data/Mooney --blur B --white W
python -m mooney_tool.cli -i data/Original -o data/variants/inverted    --blur B --white W --invert
python -m mooney_tool.cli -i data/Original -o data/variants/upside_down --blur B --white W --upside-down
```

Core (`mooney_tool/core.py`): grayscale -> Gaussian blur (sigma px) -> threshold at the `(100 - white%)` percentile, so exactly `white%` of pixels are white. `params.csv` logs the achieved % per image.

## 3. Vision system 1: Qwen3-VL (own env)

```bash
python3 -m venv .venv-vlm && source .venv-vlm/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
pip install "transformers>=4.57" accelerate pillow opencv-python
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"   # must include sm_120

python vision/run_vlm.py --images data/Original --out outputs/Result1/Original
python vision/run_vlm.py --images data/Mooney   --out outputs/Result1/Mooney
python vision/run_vlm.py --images data/variants/inverted    --out outputs/variants/Result1/inverted
python vision/run_vlm.py --images data/variants/upside_down --out outputs/variants/Result1/upside_down
```

## 4. Vision system 2: SAM 3 (own env, Python 3.12)

```bash
uv venv .venv-sam3 --python 3.12 && source .venv-sam3/bin/activate      # or: conda create -n sam3 python=3.12
uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3 ../sam3 && uv pip install -e ../sam3 einops psutil "setuptools<80" pycocotools scikit-image scipy "opencv-python<4.12"   # sam3's pyproject misses these
hf auth login        # account with access to facebook/sam3

python vision/run_sam3.py --images data/Original --out outputs/Result2/Original
python vision/run_sam3.py --images data/Mooney   --out outputs/Result2/Mooney
python vision/run_sam3.py --images data/variants/inverted    --out outputs/variants/Result2/inverted
python vision/run_sam3.py --images data/variants/upside_down --out outputs/variants/Result2/upside_down --flip-gt
```

Run the two systems one after the other (12 GB is not enough for both at once).

## 5. Optional add-on: Gemini on the same probes (slides only)

Same prompts, one fresh stateless request per probe (no chat history). Needs `pip install google-genai` and `export GEMINI_API_KEY=...`; use a model id from Google AI Studio. Free-tier quotas are small, so `--probes name,pick` keeps it to 2 requests per image.

```bash
python vision/run_vlm.py --backend gemini --model <model-id> --probes name,pick \
    --images data/Mooney --out outputs/extra/gemini/Mooney --tag gemini_Mooney
```

## 6. Package the submission

```bash
python scripts/package.py --pptx MooneyMinded_Part1.pptx
# -> submission/MooneyMinded_Part1.zip + .md5; checks all counts, exits non-zero if anything is missing
```

## Outputs

- `outputs/Result{1,2}/{Original,Mooney}/`: per-image results for the submission (JSON answers / mask overlays).
- `results/*.csv`: scored tables (small, committed) used for the analysis and slides.
- Images, masks and weights are git-ignored (`data/`, `outputs/`): the repo is public and the photos are COCO's.
