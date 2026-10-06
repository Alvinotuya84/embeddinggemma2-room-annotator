"""Find which export stage changes the image model's numbers (export.py check 3 failing).

Each stage is compared against the eager PyTorch wrapper, which check 2 already
proved matches the original model. Run the cheap stages first:

  python diagnose.py --stages export portable      # ~10-20 min
  python diagnose.py --stages vision-xnnpack       # ~20 min, vision half only
  python diagnose.py --stages text-xnnpack         # ~20 min, text half only

Reading the results:
  export low                -> torch.export traced something differently
  export ok, portable low   -> ExecuTorch's edge passes or portable kernels
  portable ok               -> XNNPACK; the two *-xnnpack stages say which half
"""

import argparse
import gc
import pathlib

import torch

import export as E


class VisionHalf(torch.nn.Module):
    def __init__(self, image_module):
        super().__init__()
        self.m = image_module

    def forward(self, image):
        return self.m.soft_tokens(image)


class TextHalf(torch.nn.Module):
    def __init__(self, image_module):
        super().__init__()
        self.m = image_module

    def forward(self, soft):
        return self.m.text_embedding(soft)


def report(name, got, want):
    c = E.cosine(got, want)
    diff = (got.double() - want.double()).abs().max().item()
    verdict = "ok" if c >= E.MIN_COSINE else "DRIFTS"
    print(f"  {name}: cosine {c:.6f}, max abs diff {diff:.3g}  -> {verdict}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=int, choices=sorted(E.SIDE_TO_BUDGET), default=384)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["export", "portable", "vision-xnnpack", "text-xnnpack"],
        default=["export", "portable"],
    )
    parser.add_argument("--out", default="diag")
    args = parser.parse_args()

    from executorch.exir import to_edge_transform_and_lower
    from transformers import AutoModel, AutoProcessor

    E.speed_up_fx_provenance()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model ...", flush=True)
    model = AutoModel.from_pretrained(
        E.MODEL_ID, dtype=torch.float32, attn_implementation="sdpa", audio_config=None
    ).eval()
    processor = AutoProcessor.from_pretrained(E.MODEL_ID)
    img = E.test_image(args.side, None)
    x = E.to_tensor(img).contiguous()

    with torch.no_grad():
        _, template = E.reference_image_embedding(model, processor, img, E.SIDE_TO_BUDGET[args.side], 768)
        module = E.ImageEmbedder(model, template, model.config.image_token_id, args.side, 768).eval()
        module.text.embed_tokens = torch.nn.Identity()
        gc.collect()

        eager = module(x)
        soft = module.soft_tokens(x)
        print(f"Eager soft tokens {tuple(soft.shape)}, |max| {soft.abs().max().item():.3g}", flush=True)

        if "export" in args.stages or "portable" in args.stages:
            print("Stage: torch.export", flush=True)
            ep = torch.export.export(module, (x,), strict=False)
            report("exported program", ep.module()(x), eager)

            if "portable" in args.stages:
                print("Stage: ExecuTorch edge dialect (no backend)", flush=True)
                edge = to_edge_transform_and_lower(ep)
                report("edge program (Python)", edge.exported_program().module()(x), eager)
                del ep
                gc.collect()
                print("Stage: ExecuTorch runtime, portable kernels", flush=True)
                path = out_dir / "portable.pte"
                with open(path, "wb") as f:
                    edge.to_executorch().write_to_file(f)
                del edge
                gc.collect()
                report("portable .pte", E.run_pte(path, [x]), eager)

        if "vision-xnnpack" in args.stages:
            print("Stage: vision half through XNNPACK", flush=True)
            path = out_dir / "vision_xnnpack.pte"
            E.lower_to_pte(VisionHalf(module).eval(), (x,), path)
            report("vision .pte soft tokens", E.run_pte(path, [x]), soft)

        if "text-xnnpack" in args.stages:
            print("Stage: text half through XNNPACK", flush=True)
            path = out_dir / "text_xnnpack.pte"
            E.lower_to_pte(TextHalf(module).eval(), (soft,), path)
            report("text .pte embedding", E.run_pte(path, [soft]), eager)


if __name__ == "__main__":
    main()
