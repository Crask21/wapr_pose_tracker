from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from wapr_pose_tracker.sam_compat import PromptAlignedSAM


class SAMCompatibilityTests(unittest.TestCase):
    def test_filtered_masks_are_restored_to_original_prompt_rows(self):
        data = torch.zeros((2, 12, 16), dtype=torch.bool)
        data[0, 1:4, 2:5] = True
        data[1, 6:9, 9:12] = True
        result = SimpleNamespace(masks=SimpleNamespace(data=data),
                                 boxes=SimpleNamespace(cls=torch.tensor([2., 0.])))
        segmentor = Mock()
        segmentor.predict.return_value = [result]
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        boxes = [[0, 0, 16, 12]] * 3
        out = PromptAlignedSAM(segmentor).predict(image, bboxes=boxes, device="cpu", verbose=False)[0].masks.data
        self.assertEqual(tuple(out.shape), (3, 12, 16))
        self.assertTrue(torch.equal(out[2], data[0]))
        self.assertTrue(torch.equal(out[0], data[1]))
        self.assertFalse(out[1].any())
        segmentor.predict.assert_called_once_with(image, bboxes=boxes, device="cpu", verbose=False)

    def test_empty_results_produce_empty_prompt_masks(self):
        segmentor = Mock()
        segmentor.predict.return_value = [SimpleNamespace(masks=None)]
        out = PromptAlignedSAM(segmentor).predict(np.zeros((12, 16, 3)), bboxes=[[0, 0, 5, 5]], device="cpu")[0]
        self.assertEqual(tuple(out.masks.data.shape), (1, 12, 16))
        self.assertFalse(out.masks.data.any())

    def test_invalid_mapping_and_resolution_are_rejected(self):
        segmentor = Mock()
        for indices, shape in (([0., 0.], (2, 12, 16)), ([0., 3.], (2, 12, 16)),
                               ([0., 0.5], (2, 12, 16)), ([0., 1.], (2, 24, 32))):
            segmentor.predict.return_value = [SimpleNamespace(
                masks=SimpleNamespace(data=torch.ones(shape)),
                boxes=SimpleNamespace(cls=torch.tensor(indices)),
            )]
            with self.assertRaises(ValueError):
                PromptAlignedSAM(segmentor).predict(np.zeros((12, 16, 3)), bboxes=[[0, 0, 5, 5]] * 2)


if __name__ == "__main__":
    unittest.main()
