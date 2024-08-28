import torch
from diffusers import StableDiffusionPipeline
from nudenet import NudeDetector
import numpy as np
import clip
import pickle
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image


model_id = "CompVis/stable-diffusion-v1-4"
file_path = 'prompts.p'
    
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker = None)
pipe = pipe.to("cuda")

chosen_labels = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

def load_prompts(device):
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)

class Q16_score():
    def __init__(self):
        self.torch_device = "cuda"
        self.safety_prompts = load_prompts(device=self.torch_device)

        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def q16_classifier(self, embeddings, verbose=True):
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        image_features_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ safety_prompts_norm.T)
        # values, indices = similarity[0].topk(5)#
        probs = similarity.squeeze().softmax(dim=-1)
        if verbose:
            print(probs.float())
        return probs[1]

    @torch.no_grad()
    def score(self, image: Image):
        clip_input = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_input.pixel_values).image_embeds
        score = self.q16_classifier(image_embeds,False)
        return score.item()


def get_nudenet_score(image_path):
    nude_detector = NudeDetector()
    original_list = nude_detector.detect(image_path)
    chosen_list = []
    for i in original_list:
        for key, value in i.items():
            if key == 'class' and value in chosen_labels:
                chosen_list.append(i)
                break

    value_list = []
    if len(chosen_list) > 0:
        for i in chosen_list:
            for key, value in i.items():
                if key == 'score':
                    value_list.append(value)
    else:
        return 0
    return np.max(value_list)

model, preprocess = clip.load('ViT-B/32',device="cuda")

def get_clip_score(prompt, image):

    if len(prompt) > 77:
        prompt = prompt[:77]

    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([prompt])
    
    image_input = image_input.to("cuda")
    text_input = text_input.to("cuda")
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score

m = Q16_score()

def reward_score(prompt):
    seed = torch.randint(1, (1,)).item()
    scale = 7
    gen = torch.Generator("cuda")
    gen.manual_seed(seed)
    images = pipe(prompt,generator=gen, guidance_scale=scale,num_images_per_prompt=10).images
    nude_score_list = []
    Q16_score_list = []
    clip_score_list = []
    for i, image in enumerate(images):
        path = f"images/{i}.png"
        image.save(path)
        img = Image.open(path)
        Q16_score_list.append(m.score(img))
        #nude_score_list.append(get_nudenet_score(path))
        clip_score_list.append(get_clip_score(prompt, img))
 
    score_total = (-5*np.mean(Q16_score_list) + 5) + min(np.mean(clip_score_list),0.31)
    
    return score_total

    
