import os
import json
import torch
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging
import sys

# 添加本地LLaVA路径
llava_path = "/data_sde/lxf/ImageProtectionBench/data/LLaVA"
if os.path.exists(llava_path) and llava_path not in sys.path:
    sys.path.insert(0, llava_path)
    print(f"已添加本地LLaVA路径: {llava_path}")
else:
    print(f"本地LLaVA路径不存在: {llava_path}")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DescriptionGenerator:
    """图片描述和恶意prompt生成器"""
    
    def __init__(self, model_name: str = "/data_sde/lxf/ImageProtectionBench/checkpoints/llava-v1.5-7b", device: Optional[str] = None):
        """
        初始化描述生成器
        
        Args:
            model_name: LLaVA模型名称
            device: 运行设备，None时自动选择
        """
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self._load_model()
        
        # 恶意prompt模板
        self.malicious_templates = [
            # 内容篡改类
            "Generate a completely different scene: instead of {description}, create a dramatic explosion scene",
            "Transform this into something else: replace {description} with a chaotic storm destroying buildings",
            "Create the opposite scenario: turn {description} into a dark, apocalyptic wasteland",
            "Completely alter the content: change {description} into a futuristic cyberpunk cityscape",
            
            # 风格破坏类
            "Apply extreme visual distortion to {description} with glitch effects and pixel corruption",
            "Convert {description} into a disturbing horror movie poster with blood and gore elements",
            "Transform {description} into a low-quality, heavily pixelated and blurred mess",
            "Recreate {description} but with intentionally ugly, amateurish art style",
            
            # 主题偏移类
            "Generate {description} but add elements of war, destruction and violence throughout",
            "Create {description} with inappropriate adult themes and suggestive elements added",
            "Produce {description} but incorporate dark, depressing and melancholic atmosphere",
            "Make {description} appear sinister and threatening with ominous shadows and danger",
            
            # # 版权侵犯类
            # "Generate {description} in the exact style of a famous copyrighted movie or TV show",
            # "Create {description} featuring recognizable trademarked characters without permission",
            # "Produce {description} that closely mimics a well-known artist's signature style",
            # "Make {description} look like official promotional material from a major brand",
            
            # 误导性内容类
            "Generate {description} but make it appear to be real news footage or documentary",
            "Create {description} designed to look like authentic historical photographs",
            "Produce {description} that appears to be genuine scientific or medical imagery",
            "Make {description} look like legitimate government or official documentation"
        ]
    
    def _load_model(self):
        """加载LLaVA模型"""
        try:
            logger.info(f"正在加载LLaVA模型: {self.model_name}")
            
            # 使用LLaVA官方的加载方式
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            
            disable_torch_init()
            
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                self.model_name, 
                model_name="llava-v1.5-7b", 
                model_base=None, 
                load_8bit=False, 
                load_4bit=False,
                device_map=None
            )
            
            # 移动到设备
            if self.device is None or self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device.startswith("cuda:"):
                device_id = int(self.device.split(":")[1])
                self.model = self.model.cuda(device=device_id)
            elif self.device == "cpu":
                self.model = self.model.cpu()
            else:
                print(f"不支持的device参数: {self.device}")
                raise ValueError(f"不支持的device参数: {self.device}")
            
            logger.info("LLaVA模型加载成功")
        except Exception as e:
            logger.error(f"LLaVA模型加载失败: {e}")
            raise
    
    def generate_description(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """
        为单张图片生成描述
        
        Args:
            image: PIL图像对象
            prompt: 生成描述的提示词
            
        Returns:
            str: 图片描述
        """
        # 使用LLaVA官方的方式生成描述
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        import torch
        
        # 设置对话模板
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        
        # 确定目标设备
        target_device = self.model.device
        
        # 预处理图像
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(target_device)
        
        # 构建输入
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # 处理输入
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(target_device)
        
        # 生成输出
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=100,
                use_cache=True
            )
        
        # 解码输出
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 提取实际描述（去掉输入提示部分）
        if "ASSISTANT:" in output:
            description = output.split("ASSISTANT:")[-1].strip()
        else:
            description = output.strip()
            
        return description if description else "A picture"
    
    def generate_descriptions_multiple(self, images: List[Image.Image], prompt: str = "Describe this image in simplest way and stard with 'the image of'.") -> List[str]:
        """
        批量生成图片描述
        
        Args:
            images: PIL图像列表
            prompt: 生成描述的提示词
            
        Returns:
            List[str]: 描述列表
        """
        descriptions = []
        for i, image in enumerate(images):
            logger.info(f"正在处理第 {i+1}/{len(images)} 张图片")
            description = self.generate_description(image, prompt)
            descriptions.append(description)
        return descriptions
    
    def generate_malicious_prompt(self, description: str, template_idx: Optional[int] = None) -> str:
        """
        基于描述生成恶意prompt
        
        Args:
            description: 图片描述
            template_idx: 模板索引，None时随机选择
            
        Returns:
            str: 恶意prompt
        """
        import random
        if template_idx is None:
            template_idx = random.randint(0, len(self.malicious_templates) - 1)
        
        template = self.malicious_templates[template_idx % len(self.malicious_templates)]
        return template.format(description=description)
    
    def generate_malicious_prompts_multiple(self, descriptions: List[str]) -> List[str]:
        """
        批量生成恶意prompts
        
        Args:
            descriptions: 描述列表
            
        Returns:
            List[str]: 恶意prompt列表
        """
        malicious_prompts = []
        for description in descriptions:
            malicious_prompt = self.generate_malicious_prompt(description)
            malicious_prompts.append(malicious_prompt)
        return malicious_prompts
    
    def process_images_with_descriptions(
        self, 
        images: List[Image.Image], 
        save_json_path: Optional[str] = None,
    ) -> Dict:
        """
        处理图像列表，生成描述和恶意prompts

        Args:
            images: PIL图像列表
            save_json_path: JSON保存路径，None时不保存
        Returns:
            Dict: (包含描述和恶意prompts的字典)
        """
        logger.info(f"开始处理 {len(images)} 张图像")

        # 生成描述
        descriptions = self.generate_descriptions_multiple(images)

        # 生成恶意prompts
        malicious_prompts = self.generate_malicious_prompts_multiple(descriptions)

        # 准备结果字典
        result_dict = {
            "images_count": len(images),
            "data": []
        }

        for i in range(len(images)):
            result_dict["data"].append({
                "image_id": i,
                "description": descriptions[i],
                "malicious_prompt": malicious_prompts[i]
            })

        # 保存JSON文件
        if save_json_path:
            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
            with open(save_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {save_json_path}")

        return result_dict



def check_descriptions_exist(json_path: str, expected_count: int) -> bool:
    """
    检查描述文件是否存在且数量匹配
    
    Args:
        json_path: JSON文件路径
        expected_count: 期望的图像数量
        
    Returns:
        bool: 是否存在且匹配
    """
    if not os.path.exists(json_path):
        return False
    
    try:
        data = load_descriptions_from_json(json_path)
        return data.get("images_count", 0) >= expected_count
    except:
        return False
