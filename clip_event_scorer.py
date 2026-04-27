# clip_event_scorer.py
# UTF-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from PIL import Image

import clip


@dataclass
class ClipEventResult:
    label: str
    scores: Dict[str, float]


class ClipEventScorer:
    def __init__(self, device: str = "") -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.prompt_groups: Dict[str, List[str]] = {
            "neutral": [
                "a dog standing on grass",
                "a dog walking on grass",
                "a dog sniffing the ground",
                "a dog sitting outside",
                "a dog",
            ],
            "poop": [
                "a dog pooping on grass",
                "a dog defecating outside",
                "a dog squatting to poop",
                "poop visibly leaving a dog"
            ],
            "pee": [
                "a dog peeing on grass",
                "a dog urinating outside",
                "a dog lifting a leg to pee",
                "a dog squatting to pee",
                "urine visibly leaving a dog",
            ],
        }

        self.class_names: List[str] = list(self.prompt_groups.keys())
        flat_prompts: List[str] = []
        self.class_ranges: Dict[str, Tuple[int, int]] = {}

        start = 0
        for class_name in self.class_names:
            prompts = self.prompt_groups[class_name]
            flat_prompts.extend(prompts)
            end = start + len(prompts)
            self.class_ranges[class_name] = (start, end)
            start = end

        with torch.no_grad():
            text_tokens = clip.tokenize(flat_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

    def score_pil(self, pil_image: Image.Image) -> ClipEventResult:
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ self.text_features.T
            prompt_probs = logits.softmax(dim=-1).squeeze(0)

        class_scores: Dict[str, float] = {}
        for class_name in self.class_names:
            start, end = self.class_ranges[class_name]
            class_scores[class_name] = float(prompt_probs[start:end].mean().item())

        best_label = max(class_scores, key=class_scores.get)
        return ClipEventResult(label=best_label, scores=class_scores)

    def score_bgr_crop(self, crop_bgr) -> ClipEventResult:
        pil_image = Image.fromarray(crop_bgr[:, :, ::-1])
        return self.score_pil(pil_image)