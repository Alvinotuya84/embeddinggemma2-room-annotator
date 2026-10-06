"""Export EmbeddingGemma 2 to ExecuTorch `.pte` files for react-native-executorch.

Produces files that plug straight into the library's built-in pipelines:

  embeddinggemma2_image_<side>.pte   useImageEmbedder: f32[1,3,S,S] (RGB, 0..1) -> f32[1,D]
  embeddinggemma2_text.pte           useTextEmbedder:  i64[1,L] ids, i64[1,L] mask -> f32[1,D]
  tokenizer.json                     for useTextEmbedder

Why a custom wrapper: the Hugging Face forward pass cuts images into patches in
the processor and strips padding with data-dependent ops (`.tolist()`,
`torch.split`, boolean-mask indexing, `masked_scatter`), none of which
`torch.export` can trace. The camera crop is always a fixed-size square with no
padding, so everything that depends only on the image size (patch positions,
the 3x3 pooling matrix, the token template around the image) is precomputed
here as constants, and the graph keeps only the real computation.

Every stage is checked against the reference model before anything is written,
and the written `.pte` is run with the ExecuTorch runtime and checked again.
"""

import argparse
import gc
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "google/embeddinggemma-2"
PATCH = 16
POOL = 3
# Square side -> processor `max_soft_tokens` budget whose aspect-preserving resize
# leaves an image of exactly this side unchanged (see Gemma4ImageProcessor).
SIDE_TO_BUDGET = {384: 70, 768: 280}
MAX_TEXT_LEN = 512  # Equal to the sliding window, so sliding layers behave like full attention.
MIN_COSINE = 0.999


def patchify(images: torch.Tensor) -> torch.Tensor:
    """[B,3,S,S] -> [B, (S/16)^2, 16*16*3], in the same order as the processor's convert_image_to_patches."""
    b, c, h, w = images.shape
    ph, pw = h // PATCH, w // PATCH
    x = images.reshape(b, c, ph, PATCH, pw, PATCH).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, ph * pw, PATCH * PATCH * c)


def patch_positions(side: int) -> torch.Tensor:
    """(x, y) per patch, [1, P, 2], matching the processor's meshgrid(indexing="xy")."""
    n = side // PATCH
    grid = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="xy")
    return torch.stack(grid, dim=-1).reshape(1, n * n, 2)


class ImageEmbedder(torch.nn.Module):
    """Square RGB image in [0, 1] -> normalized embedding."""

    def __init__(self, model, template_ids: torch.Tensor, image_token_id: int, side: int, dim: int):
        super().__init__()
        vision = model.vision_tower
        text = model.language_model
        self.dim = dim
        self.input_proj = vision.patch_embedder.input_proj
        self.encoder = vision.encoder
        self.embed_vision = model.embed_vision
        self.text = text
        self.root_hidden_size = vision.pooler.root_hidden_size
        self.standardize = bool(vision.config.standardize)
        if self.standardize:
            self.register_buffer("std_bias", vision.std_bias.float().clone())
            self.register_buffer("std_scale", vision.std_scale.float().clone())

        positions = patch_positions(side)
        num_patches = positions.shape[1]
        num_soft = num_patches // POOL**2
        padding = torch.zeros(1, num_patches, dtype=torch.bool)

        with torch.no_grad():
            # Patch position embeddings depend only on positions: precompute them.
            self.register_buffer("pos_emb", vision.patch_embedder._position_embeddings(positions, padding).clone())
            # The 3x3 spatial pooling is a fixed linear map for a fixed grid: W^T @ I = W^T.
            eye = torch.eye(num_patches).unsqueeze(0)
            pool_t, _ = vision.pooler._avg_pool_by_positions(eye, positions, num_soft)
            self.register_buffer("pool_t", pool_t.float().clone())  # [1, N, P]

            # Text embeddings for the tokens around the image (bos, boi, eoi, ...).
            image_mask = template_ids[0] == image_token_id
            idx = image_mask.nonzero().flatten()
            if len(idx) != num_soft or not torch.equal(idx, torch.arange(idx[0], idx[0] + num_soft)):
                raise ValueError(f"Expected {num_soft} contiguous image tokens, found {len(idx)}")
            start = int(idx[0])
            ids = torch.where(image_mask, model.config.text_config.pad_token_id, template_ids)
            embeds = text.embed_tokens(ids)
            self.register_buffer("prefix", embeds[:, :start].clone())
            self.register_buffer("suffix", embeds[:, start + num_soft :].clone())

        self.register_buffer("positions", positions)
        self.register_buffer("valid", ~padding)

    def soft_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Vision half: image -> [1, N, text_hidden] soft tokens."""
        # Same as Gemma4VisionPatchEmbedder.forward, with precomputed position embeddings.
        pixels = 2 * (patchify(image) - 0.5)
        hidden = self.input_proj(pixels) + self.pos_emb
        hidden = self.encoder(
            inputs_embeds=hidden, attention_mask=self.valid, pixel_position_ids=self.positions
        ).last_hidden_state
        # Same as Gemma4VisionPooler + standardize, without padding removal (there is none).
        soft = (self.pool_t @ hidden.float()) * self.root_hidden_size
        if self.standardize:
            soft = (soft - self.std_bias) * self.std_scale
        return self.embed_vision(inputs_embeds=soft.to(hidden.dtype)[0]).unsqueeze(0)

    def text_embedding(self, soft: torch.Tensor) -> torch.Tensor:
        """Text half: soft tokens -> normalized embedding."""
        embeds = torch.cat([self.prefix, soft.to(self.prefix.dtype), self.suffix], dim=1)
        out = self.text(inputs_embeds=embeds).last_hidden_state
        return F.normalize(out.mean(dim=1)[..., : self.dim], dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.text_embedding(self.soft_tokens(image))


class TextEmbedder(torch.nn.Module):
    """Token ids (prompt already prepended by the app) -> normalized embedding."""

    def __init__(self, model, dim: int):
        super().__init__()
        self.text = model.language_model
        self.dim = dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled[..., : self.dim], dim=-1)


def reference_image_embedding(model, processor, image: Image.Image, budget: int, dim: int):
    """Unmodified Hugging Face path: processor -> model -> mean pool -> normalize."""
    inputs = processor(
        text=[processor.image_token], images=[[image]], max_soft_tokens=budget, return_tensors="pt"
    )
    model_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_position_ids")}
    out = model(**model_inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(out.dtype)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return F.normalize(pooled[..., :dim], dim=-1), inputs["input_ids"]


def test_image(side: int, path: str | None) -> Image.Image:
    if path:
        img = Image.open(path).convert("RGB")
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        return img.crop((left, top, left + s, top + s)).resize((side, side), Image.BICUBIC)
    # Structured synthetic scene; random noise is a poor test for a vision model.
    img = Image.new("RGB", (side, side), (200, 190, 170))
    draw = ImageDraw.Draw(img)
    draw.rectangle([side * 0.15, side * 0.2, side * 0.85, side * 0.65], fill=(20, 20, 25))
    draw.rectangle([side * 0.45, side * 0.65, side * 0.55, side * 0.8], fill=(60, 60, 60))
    draw.ellipse([side * 0.05, side * 0.75, side * 0.3, side * 0.95], fill=(150, 40, 40))
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def check(name: str, a: torch.Tensor, b: torch.Tensor):
    c = cosine(a, b)
    print(f"  {name}: cosine {c:.6f}")
    if c < MIN_COSINE:
        raise SystemExit(f"FAILED: {name} below {MIN_COSINE}")


def speed_up_fx_provenance():
    """Back-port of a PyTorch fix that makes lowering large graphs hours faster.

    Older torch versions deep-copy every node's provenance history
    (`node.meta["from_node"]`) on every pass, which grows quadratically on a
    graph this size. Newer torch shares the records instead (they are immutable
    once created). This only affects debug metadata, never the computation.
    """
    import copy
    import types

    import torch.fx.traceback as fx_traceback

    if getattr(fx_traceback, "copy", None) is not copy:
        return  # Already patched, or torch no longer uses copy here.

    def deepcopy(x, memo=None):
        if isinstance(x, list) and all(isinstance(i, fx_traceback.NodeSource) for i in x):
            return list(x)
        return copy.deepcopy(x, memo)

    shim = types.ModuleType("copy")
    shim.__dict__.update(copy.__dict__)
    shim.deepcopy = deepcopy
    fx_traceback.copy = shim


def lower_to_pte(module, args, path: pathlib.Path, dynamic_shapes=None, schema=None, xnnpack=True):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    print("  torch.export ...")
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    print("  lowering to XNNPACK ..." if xnnpack else "  lowering (portable kernels, no XNNPACK) ...")
    constant_methods = {"get_model_schema": json.dumps(schema)} if schema else None
    partitioner = [XnnpackPartitioner()] if xnnpack else []
    edge = to_edge_transform_and_lower(ep, partitioner=partitioner, constant_methods=constant_methods)
    # Each stage holds its own copy of the weights; drop them as soon as they are not needed.
    del ep
    gc.collect()
    print("  serializing ...")
    program = edge.to_executorch()
    del edge
    gc.collect()
    with open(path, "wb") as f:
        program.write_to_file(f)
    del program
    gc.collect()
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


def run_pte(path: pathlib.Path, inputs: list[torch.Tensor]) -> torch.Tensor:
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(path).load_method("forward")
    # The runtime expects dense, contiguous CPU tensors (to_tensor() returns a permuted view).
    return method.execute([t.contiguous() for t in inputs])[0]


def text_schema(dim: int) -> dict:
    seq = {"kind": "range", "range": {"min": 2, "max": MAX_TEXT_LEN, "step": 1}}
    ids = {"kind": "Tensor", "dtype": "int64", "shape": [{"kind": "constant", "value": 1}, seq]}
    return {
        "forward": {
            "inputs": [ids, ids],
            "outputs": [
                {
                    "kind": "Tensor",
                    "dtype": "float32",
                    "shape": [{"kind": "constant", "value": 1}, {"kind": "constant", "value": dim}],
                }
            ],
            "runtimeConstraints": [
                {
                    "kind": "equality",
                    "dims": [
                        {"paramSide": "input", "tensorIdx": 0, "dimIdx": 1},
                        {"paramSide": "input", "tensorIdx": 1, "dimIdx": 1},
                    ],
                }
            ],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=int, choices=sorted(SIDE_TO_BUDGET), default=384,
                        help="Square image size the app feeds in (384 = 64 image tokens, 768 = 256).")
    parser.add_argument("--dim", type=int, choices=(768, 512, 256, 128), default=768,
                        help="Matryoshka output size. 256 loses little quality and shrinks stored vectors 3x.")
    parser.add_argument("--skip-text", action="store_true", help="Only export the image model.")
    parser.add_argument("--test-image", help="Optional photo used for the accuracy checks.")
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--skip-st-check", action="store_true",
                        help="Skip check 1 (needs a second copy of the model in memory) once it has passed.")
    parser.add_argument("--push-to-hub", metavar="REPO_ID",
                        help="Upload the output to this Hugging Face model repo (needs a write token).")
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    speed_up_fx_provenance()

    print(f"Loading {MODEL_ID} (float32, no audio tower) ...")
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="sdpa", audio_config=None
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    image_token_id = model.config.image_token_id

    with torch.no_grad():
        # 1. Our reference path must match sentence-transformers (validates the token template).
        if args.skip_st_check:
            print("Check 1/3: skipped (--skip-st-check)")
        else:
            from sentence_transformers import SentenceTransformer

            print("Check 1/3: Hugging Face reference vs sentence-transformers (768 px, default budget)")
            st = SentenceTransformer(MODEL_ID, config_kwargs={"audio_config": None})
            img768 = test_image(768, args.test_image)
            st_emb = torch.from_numpy(np.asarray(st.encode([{"image": img768}], normalize_embeddings=True)))
            ref768, _ = reference_image_embedding(model, processor, img768, SIDE_TO_BUDGET[768], 768)
            check("reference vs sentence-transformers", ref768, st_emb)
            del st
            gc.collect()

        img = test_image(args.side, args.test_image)
        ref_img, template_ids = reference_image_embedding(model, processor, img, SIDE_TO_BUDGET[args.side], args.dim)
        print(f"  image template: {template_ids.shape[1]} tokens")

        text_inputs = processor.tokenizer(["task: search result | query: Alvin's TV"], return_tensors="pt")

        # 2. Text export (before the image wrapper, which drops the 134M-param token table).
        if not args.skip_text:
            print("Text model")
            text_module = TextEmbedder(model, args.dim).eval()
            seq = torch.export.Dim("seq", min=2, max=MAX_TEXT_LEN)
            text_path = out_dir / "embeddinggemma2_text.pte"
            lower_to_pte(
                text_module,
                (text_inputs["input_ids"], text_inputs["attention_mask"]),
                text_path,
                dynamic_shapes={"input_ids": {1: seq}, "attention_mask": {1: seq}},
                schema=text_schema(args.dim),
            )
            processor.tokenizer.save_pretrained(out_dir / "tokenizer")
            (out_dir / "tokenizer" / "tokenizer.json").rename(out_dir / "tokenizer.json")
            print("Check 3/3 (text): .pte vs PyTorch")
            longer = processor.tokenizer(["title: none | text: a black television on a wooden stand"], return_tensors="pt")
            check("text .pte (other length)", run_pte(text_path, [longer["input_ids"], longer["attention_mask"]]),
                  TextEmbedder(model, args.dim)(longer["input_ids"], longer["attention_mask"]))

        # 3. Image wrapper must match the reference path.
        print(f"Check 2/3: export wrapper vs reference ({args.side} px)")
        image_module = ImageEmbedder(model, template_ids, image_token_id, args.side, args.dim).eval()
        image_module.text.embed_tokens = torch.nn.Identity()  # Template embeddings are baked in.
        gc.collect()  # Frees the 134M-param token table (about 540 MB).
        x = to_tensor(img)
        check("wrapper vs reference", image_module(x), ref_img)

        print("Image model")
        image_path = out_dir / f"embeddinggemma2_image_{args.side}.pte"
        lower_to_pte(image_module, (x,), image_path)
        print("Check 3/3 (image): .pte vs reference")
        check("image .pte", run_pte(image_path, [x]), ref_img)

    print("\nDone. Files in", out_dir.resolve())

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        commit = api.upload_folder(
            folder_path=out_dir, repo_id=args.push_to_hub, allow_patterns=["*.pte", "tokenizer.json"]
        )
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")
        print(f"Commit (pin this in the app's model-config.ts): {commit.oid}")


if __name__ == "__main__":
    main()
