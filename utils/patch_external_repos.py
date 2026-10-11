# utils/patch_external_repos.py
"""
Patches a fresh clone of external/SAM-Med3D. Its pos_embed/mask_decoder code
assumes 128^3 inputs (the pretraining size), but we use 64^3 patches, so the
embeddings are interpolated to the input size; otherwise it crashes with a
shape error.

I first had this as a local edit in the cloned repo, which got lost on every
fresh clone (e.g. on Kaggle), so it lives here now. If upstream changes and the
original text isn't found, it raises an error instead of silently skipping.

Run standalone (`python -m utils.patch_external_repos`) or import
`patch_sam_med3d(root)` and call it right after cloning.
"""

import os

_POS_EMBED_ORIGINAL = """        if self.pos_embed is not None:
            x = x + self.pos_embed
"""

_POS_EMBED_PATCHED = """        if self.pos_embed is not None:
            if self.pos_embed.shape[1:4] != x.shape[1:4]:
                pos_embed = F.interpolate(
                    self.pos_embed.permute(0, 4, 1, 2, 3),
                    size=x.shape[1:4],
                    mode="trilinear",
                    align_corners=False
                ).permute(0, 2, 3, 4, 1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed
"""

_MASK_DECODER_ORIGINAL = """        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
"""

_MASK_DECODER_PATCHED = """        else:
            src = image_embeddings

        if dense_prompt_embeddings.shape[2:] != src.shape[2:]:
            dense_pe = F.interpolate(dense_prompt_embeddings, size=src.shape[2:], mode="trilinear", align_corners=False)
        else:
            dense_pe = dense_prompt_embeddings
        src = src + dense_pe

        if image_pe.shape[2:] != src.shape[2:]:
            image_pe = F.interpolate(image_pe, size=src.shape[2:], mode="trilinear", align_corners=False)

        if image_pe.shape[0] != tokens.shape[0]:
"""


def _apply_patch(path, original, patched, label):
    with open(path) as f:
        text = f.read()

    if patched in text:
        print(f"[*] {label}: already patched, skipping")
        return
    if original not in text:
        raise RuntimeError(
            f"{label}: neither the original nor the patched form of this fix "
            f"was found in {path} -- upstream source has likely changed and "
            f"this patch needs updating."
        )

    with open(path, "w") as f:
        f.write(text.replace(original, patched))
    print(f"[*] {label}: patched")


def patch_sam_med3d(root="./external/SAM-Med3D"):
    modeling_dir = os.path.join(root, "segment_anything", "modeling")

    _apply_patch(
        os.path.join(modeling_dir, "image_encoder3D.py"),
        _POS_EMBED_ORIGINAL, _POS_EMBED_PATCHED,
        "image_encoder3D.py pos_embed interpolation",
    )
    _apply_patch(
        os.path.join(modeling_dir, "mask_decoder3D.py"),
        _MASK_DECODER_ORIGINAL, _MASK_DECODER_PATCHED,
        "mask_decoder3D.py dense/image PE interpolation",
    )


if __name__ == "__main__":
    patch_sam_med3d()
