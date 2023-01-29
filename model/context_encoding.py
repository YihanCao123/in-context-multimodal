import torch
import torch.nn.functional as F

from transformers import CLIPVisionModel


class ContextEncoding(torch.nn.Module):
    """ Encode input images using CLIP pretrained.
    """
    def __init__(self, config):
        super().__init__()
        self.args = config
        if config.clip_model is None:
            self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.clip_model = CLIPVisionModel.from_pretrained(config.clip_model)
        
    def forward(self, images):
        """
        @args images: shape [bs, n_shots+1, 3, 224, 224]
        """
        # reshape into [bs*(n_shots+1), 3, 224, 224]
        bs, n_shots = images.shape[0], images.shape[1]
        prompt_images = images.squeeze(1)

        # shape: [bs*(n_shots+1), hidden_size(768)]
        pooled_output = self.clip_model(prompt_images)['pooler_output']

        # reshape into [bs, n_shots+1, 3, 224, 224]
        pooled_output = pooled_output.view(bs, n_shots, -1)

        return pooled_output


