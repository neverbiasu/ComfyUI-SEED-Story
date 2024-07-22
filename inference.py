import torch
import json
import re
import numpy as np

from PIL import Image, ImageDraw

class SeedStoryInferenceNode:
    def __init__(self):
        self.device = 'cuda:0'
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dict": ("MODEL_DICT",),
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": False}),
                "sequence_length": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 25,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGES",)
    FUNCTION = "run_inference"
    CATEGORY = "Inference"

    def run_inference(self, model_dict, image, text, sequence_length):
        self.tokenizer = model_dict["tokenizer"]
        self.image_transform = model_dict["image_transform"]
        self.visual_encoder = model_dict["visual_encoder"]
        self.llm = model_dict["llm"]
        self.agent_model = model_dict["agent_model"]
        self.noise_scheduler = model_dict["noise_scheduler"]
        self.vae = model_dict["vae"]
        self.unet = model_dict["unet"]
        self.adapter = model_dict["adapter"]
        self.discrete_model = model_dict["discrete_model"]

        self.BOI_TOKEN = '<img>'
        self.EOI_TOKEN = '</img>'
        self.IMG_TOKEN = '<img_{:05d}>'
        self.boi_token_id = self.tokenizer.encode(self.BOI_TOKEN, add_special_tokens=False)[0]
        self.eoi_token_id = self.tokenizer.encode(self.EOI_TOKEN, add_special_tokens=False)[0]

        image = Image.fromarray(image)
        image = image.convert('RGB')
        init_image = self.add_subtitle(image, text)

        self.agent_model.llm.base_model.model.use_kv_cache_head = False
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        image_tokens = self.BOI_TOKEN + ''.join([self.IMG_TOKEN.format(int(item)) for item in range(64)]) + self.EOI_TOKEN
        prompt = '{instruction}'.format(instruction=text + image_tokens)
        print(prompt)
        print('*' * 20)

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = [self.tokenizer.bos_token_id] + input_ids

        boi_idx = input_ids.index(self.boi_token_id)
        eoi_idx = input_ids.index(self.eoi_token_id)

        input_ids = torch.tensor(input_ids).to(self.device, dtype=torch.long).unsqueeze(0)
        ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        ids_cmp_mask[0, boi_idx + 1:eoi_idx] = True
        embeds_cmp_mask = torch.tensor([True]).to(self.device, dtype=torch.bool)

        with torch.no_grad():
            image_embeds = self.visual_encoder(image_tensor)
        output = self.agent_model.generate(tokenizer=self.tokenizer,
                                           input_ids=input_ids,
                                           image_embeds=image_embeds,
                                           embeds_cmp_mask=embeds_cmp_mask,
                                           ids_cmp_mask=ids_cmp_mask,
                                           max_new_tokens=500,
                                           num_img_gen_tokens=64)
        text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()

        images = []
        images.append(self.pil_to_np(init_image))

        story_len = sequence_length
        window_size = 8
        text_id = 1
        while output['has_img_output'] and image_embeds.shape[0] < story_len:
            image_embeds_gen = output['img_gen_feat']
            images_gen = self.adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50)

            original_image = images_gen[0]
            new_image = self.add_subtitle(original_image, text)
            images.append(self.pil_to_np(new_image))

            image_embeds = torch.cat((image_embeds, image_embeds_gen), dim=0)

            if text_id >= story_len - 1:
                break

            prompt = prompt + text + image_tokens
            text_id += 1

            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            while image_embeds.shape[0] > window_size:
                eoi_prompt_idx = prompt.index(self.EOI_TOKEN)
                prompt = prompt[eoi_prompt_idx + len(self.EOI_TOKEN) + len('[INST]'):]
                image_embeds = image_embeds[1:]
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

            print(prompt)
            print('*' * 20)

            input_ids = [self.tokenizer.bos_token_id] + input_ids

            boi_idx = torch.where(torch.tensor(input_ids) == self.boi_token_id)[0].tolist()
            eoi_idx = torch.where(torch.tensor(input_ids) == self.eoi_token_id)[0].tolist()

            input_ids = torch.tensor(input_ids).to(self.device, dtype=torch.long).unsqueeze(0)

            ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for i in range(image_embeds.shape[0]):
                ids_cmp_mask[0, boi_idx[i] + 1:eoi_idx[i]] = True
            embeds_cmp_mask = torch.tensor([True] * image_embeds.shape[0]).to(self.device, dtype=torch.bool)

            output = self.agent_model.generate(tokenizer=self.tokenizer,
                                               input_ids=input_ids,
                                               image_embeds=image_embeds,
                                               embeds_cmp_mask=embeds_cmp_mask,
                                               ids_cmp_mask=ids_cmp_mask,
                                               max_new_tokens=500,
                                               num_img_gen_tokens=64)
            text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
            print(output['text'])
            print('*' * 20)
        
        return images

    def pil_to_np(self, pil_img):
        return np.array(pil_img)

    def add_subtitle(self, original_image, text):
        text_height = 80
        new_image_size = (original_image.width, original_image.height + text_height)
        new_image = Image.new("RGB", new_image_size, "black")
        new_image.paste(original_image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        font_size = 14
        line1, line2 = text[:len(text) // 2], text[len(text) // 2:]
        text_position_line1 = (10, original_image.height + (text_height - font_size) // 2)
        text_color = "white"
        draw.text(text_position_line1, line1, fill=text_color)
        text_position_line2 = (10, text_position_line1[1] + font_size)
        draw.text(text_position_line2, line2, fill=text_color)
        return new_image

    def read_jsonl_to_dict(self, filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                json_object = json.loads(line)
                data.append(json_object)
        return data

NODE_CLASS_MAPPINGS = {
    "SeedStoryInferenceNode": SeedStoryInferenceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedStoryInferenceNode": "Seed Story Inference Node"
}
