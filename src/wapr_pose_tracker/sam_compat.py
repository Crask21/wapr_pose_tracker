"""Keep Ultralytics SAM masks aligned with WAPR's bounding-box prompts."""

from types import SimpleNamespace


class PromptAlignedSAM:
    """Adapt filtered SAM results to WAPR's (number_of_prompts, H, W) contract.

    Ultralytics records the original prompt index in boxes.cls after confidence
    filtering. Restore those rows; empty rows are discarded by WAPR's existing
    foreground check. This adapter supplies only the masks consumed by WAPR.
    """

    def __init__(self, segmentor):
        self.segmentor = segmentor

    def predict(self, image, *, bboxes, **kwargs):
        import torch

        results = self.segmentor.predict(image, bboxes=bboxes, **kwargs)
        if len(results) != 1:
            raise ValueError("Expected one SAM result for the initialization image")
        result = results[0]
        height, width = image.shape[:2]
        count = len(bboxes)
        if result.masks is None:
            masks = torch.zeros((count, height, width), dtype=torch.bool, device=kwargs.get("device", "cpu"))
        else:
            data = result.masks.data
            if data.ndim != 3 or tuple(data.shape[1:]) != (height, width):
                raise ValueError(f"SAM mask resolution {tuple(data.shape)} does not match image {(height, width)}")
            classes = result.boxes.cls
            indices = classes.to(device=data.device, dtype=torch.long)
            if (indices.numel() != data.shape[0] or not torch.isfinite(classes).all()
                    or not torch.equal(classes.to(data.device), indices.to(classes.dtype))
                    or (indices < 0).any() or (indices >= count).any()
                    or indices.unique().numel() != indices.numel()):
                raise ValueError("SAM returned invalid or ambiguous original prompt indices")
            masks = torch.zeros((count, height, width), dtype=torch.bool, device=data.device)
            masks[indices] = data.bool()
        return [SimpleNamespace(masks=SimpleNamespace(data=masks))]
