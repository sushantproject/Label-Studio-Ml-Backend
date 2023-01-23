import io
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
import json
import os
from utils import extract_model, download_file_from_s3, extract_archive

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped, get_local_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_S3_URI = "s3://sagemaker-project-p-sqr54jwsvwmr/pipelines-y15izn1hued7-TrainIntelClassifier-0i7SGpoIvV/output/model.tar.gz"

transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

idx_to_class = {
    0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'
}

# LOAD MODEL
model_dir = os.path.dirname(__file__)

# download model file from S3 into /tmp folder
extract_model(MODEL_S3_URI, model_dir)

def model_fn(model_dir, device):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")

    model.to(device).eval()

    return model




def inference(model_input, model):
    """
    Internal inference methods
    :param model_input: transformed model input data
    :return: inference output label
    """
    with torch.no_grad():
        prediction = model(model_input)
        prediction = F.softmax(prediction, dim=1)

    # Get the top  confidence of prediction
    confidence, cat_id = torch.topk(prediction, 1)
    label = idx_to_class[cat_id[0].item()]
    score = confidence[0].item()
    return score, label


def get_transformed_image(url):
    filepath = get_local_path(url)

    with open(filepath, mode='rb') as f:
        image = Image.open(f).convert('RGB')

    return transform(image).unsqueeze(0).to(device)


class ImageClassifierAPI(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        self.model = model_fn(model_dir, device)

    def predict(self, tasks, **kwargs):
        image_urls = [task['data'][self.value] for task in tasks]
        predictions = []
        for image_url in image_urls:
            image = get_transformed_image(image_url)
            score, predicted_label = inference(image, self.model)
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': float(score)})

        return predictions
