# EmbeddingGemma 2 → ExecuTorch export

Converts `google/embeddinggemma-2` into `.pte` files that load in the app's
built-in react-native-executorch pipelines (`useImageEmbedder`, `useTextEmbedder`).

## Run

Needs Python 3.10–3.13, about 12 GB of RAM and about 10 GB of free disk
(packages, model download, output). No GPU required.

```bash
cd tools/export-embeddinggemma2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py --side 384 --dim 768            # add --test-image photo.jpg to check on a real photo
```

Output in `out/`:

| File | Used by | Contract |
|---|---|---|
| `embeddinggemma2_image_384.pte` | `useImageEmbedder` | `f32[1,3,384,384]` RGB in 0..1 → `f32[1,768]` |
| `embeddinggemma2_text.pte` | `useTextEmbedder` | `i64[1,L]` ids + mask, L ≤ 512 → `f32[1,768]` |
| `tokenizer.json` | `useTextEmbedder` | |

## What the checks mean

The script stops if any check is below cosine 0.999:

1. Hugging Face reference vs sentence-transformers: proves the image token template is right.
2. Export wrapper vs reference: proves the precomputed patches, positions and pooling are exact.
3. `.pte` (run by the ExecuTorch runtime) vs PyTorch: proves lowering kept the numbers.

## Options

- `--side 768` uses 256 image tokens instead of 64: more detail, about 4x slower on device.
- `--dim 256` shortens vectors (Matryoshka). Quality drops little; storage and matching get 3x cheaper.
- `--skip-text` exports only the image model, which is all the camera matching needs.

## Notes

- Images take no prompt prefix (per the model card). Text queries do, e.g.
  `task: search result | query: ` — pass it as the `prompt` argument of `embed()`.
- `requirements.txt` pins ExecuTorch 1.4.1 and torch 2.13.0 on purpose. The app's
  react-native-executorch 0.10.x bundles the ExecuTorch 1.4.1 runtime, which is not
  guaranteed to load files from newer ExecuTorch versions. Bump these together with
  react-native-executorch. A mismatched torch also breaks check 3
  (`RuntimeError: tensor does not have a device`).
- Output is float32 (about 1.2 GB for the image model). Do not convert to float16:
  the model card warns its activations overflow float16.
