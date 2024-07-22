import hydra
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

class ModelLoaderNode:
    def __init__(self):
        self.device = 'cuda:0'
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer_cfg_path": ("STRING", {"default": "configs/tokenizer/clm_llama_tokenizer.yaml"}),
                "image_transform_cfg_path": ("STRING", {"default": "configs/processer/qwen_448_transform.yaml"}),
                "visual_encoder_cfg_path": ("STRING", {"default": "configs/visual_tokenizer/qwen_vitg_448.yaml"}),
                "llm_cfg_path": ("STRING", {"default": "configs/clm_models/llama2chat7b_lora.yaml"}),
                "agent_cfg_path": ("STRING", {"default": "configs/clm_models/agent_7b_sft.yaml"}),
                "adapter_cfg_path": ("STRING", {"default": "configs/detokenizer/detokenizer_sdxl_qwen_vit_adapted.yaml"}),
                "discrete_model_cfg_path": ("STRING", {"default": "configs/discrete_model/discrete_identity.yaml"}),
                "diffusion_model_path": ("STRING", {"default": "pretrained/stable-diffusion-xl-base-1.0"}),
            },
        }

    RETURN_TYPES = ("MODEL_DICT",)
    FUNCTION = "load_models"
    CATEGORY = "Model Loading"

    def load_models(self, tokenizer_cfg_path, image_transform_cfg_path, visual_encoder_cfg_path, 
                    llm_cfg_path, agent_cfg_path, adapter_cfg_path, discrete_model_cfg_path, diffusion_model_path):
        tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
        tokenizer = hydra.utils.instantiate(tokenizer_cfg)

        image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
        image_transform = hydra.utils.instantiate(image_transform_cfg)

        visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
        visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
        visual_encoder.eval().to(self.device, dtype=self.dtype)

        llm_cfg = OmegaConf.load(llm_cfg_path)
        llm = hydra.utils.instantiate(llm_cfg, torch_dtype='fp16')

        agent_model_cfg = OmegaConf.load(agent_cfg_path)
        agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
        agent_model.eval().to(self.device, dtype=self.dtype)

        noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(self.device, dtype=self.dtype)
        unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(self.device, dtype=self.dtype)

        adapter_cfg = OmegaConf.load(adapter_cfg_path)
        adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(self.device, dtype=self.dtype).eval()

        discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
        discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(self.device).eval()

        adapter.init_pipe(vae=vae, scheduler=noise_scheduler, visual_encoder=visual_encoder, 
                          image_transform=image_transform, discrete_model=discrete_model, dtype=self.dtype, device=self.device)

        return {
            "tokenizer": tokenizer,
            "image_transform": image_transform,
            "visual_encoder": visual_encoder,
            "llm": llm,
            "agent_model": agent_model,
            "noise_scheduler": noise_scheduler,
            "vae": vae,
            "unet": unet,
            "adapter": adapter,
            "discrete_model": discrete_model
        }
