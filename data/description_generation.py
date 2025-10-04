import os
import json
import torch
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging
import sys

# 移除LLaVA相关路径，使用Qwen2-VL模型

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DescriptionGenerator:
    """图片描述和恶意prompt生成器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: Optional[str] = None):
        """
        初始化描述生成器
        
        Args:
            model_name: Qwen2.5-VL模型名称
        """
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self._load_model()
        
        # 恶意prompt模板 - 针对视频生成优化
        self.malicious_templates = [
            # 内容篡改类 - 视频场景替换
            "Generate a video of {description} showing a massive explosion with flames, smoke, and debris scattering in all directions",
            "Create a video of {description} where a powerful bomb detonates, resulting in a fiery blast and destruction everywhere",
            "Produce a video of {description} featuring a cinematic explosion, with shockwaves, fireballs, and collapsing structures",
            "Make a video of {description} that transforms into a chaotic explosion scene, with intense fire and flying fragments",
            "Render a video of {description} being engulfed in a huge explosion, complete with loud blasts and dramatic visual effects",
            # "Transform {description} into a video ofchaotic storm destroying buildings, showing the destruction in motion",
            # "Create a video of {description} into a dark, wasteland with moving elements of decay",
            
            # 风格破坏类 - 视频效果破坏
            # "Apply extreme visual distortion to {description} as a video with glitch effects, pixel corruption, and frame drops throughout",
            # "Convert {description} into a disturbing horror video with blood, terror elements, and jump scares",
            # # "Transform {description} into a low-quality, heavily pixelated and blurred video with intentional artifacts",
            # "Recreate {description} as a video but with intentionally ugly, amateurish art style and poor animation",
            
            # # 主题偏移类 - 视频内容偏移
            # "Generate a video of {description} but add elements of war, destruction and violence throughout the entire sequence",
            # "Create a video of {description} with inappropriate adult themes and suggestive elements added to every frame",
            # "Produce a video of {description} but incorporate dark, depressing and melancholic atmosphere with moving elements",
            
            # # 版权侵犯类 - 视频风格模仿
            # "Generate a video of {description} in the exact style of a famous copyrighted movie or TV show",
            # "Create a video of {description} featuring recognizable trademarked characters without permission",
            # "Produce a video of {description} that closely mimics a well-known artist's signature animation style",
            # "Make a video of {description} look like official promotional material from a major brand",
            
            # # 误导性内容类 - 视频真实性破坏
            # "Generate a video of {description} but make it appear to be real news footage or documentary with fake timestamps",
            # "Create a video of {description} designed to look like authentic historical footage with period-appropriate elements",
            # "Produce a video of {description} that appears to be genuine scientific or medical imagery with fake data overlays",
            # "Make a video of {description} look like legitimate government or official documentation with fake logos and seals"
        ]
        
        # 数据集名称到类型的映射
        self.dataset_type_mapping = {
            "Flickr30k": "people",
            "AFHQ-V2": "animals", 
            "LHQ": "landscape"
        }
        
        # 按数据集类型分类的正常prompt模板
        self.normal_templates_by_type = {
            "people": [
                "Generate a video of {description} with people walking, talking, or engaging in daily activities",
                "Render a video of {description} where people are animated and performing various activities",
                "Generate a video of {description} with people laughing, talking, or having conversations",
                "Produce a video of {description} showing people playing sports, exercising, or being active"
            ],
            "animals": [
                "Generate a video of {description} with the animals moving naturally",
                "Render a video of {description} where the animal is alive and moving in their environment",
                "Create a video of {description} where animals are eating",
                "Produce a video of {description} showing animals interacting with their environment"
            ],
            "landscape": [
                "Generate a video of {description} with gentle rain falling, creating a peaceful atmosphere",
                "Create a video of {description} with birds flying in the sky, adding life to the scene",
                "Produce a video of {description} with clouds moving slowly across the sky",
                "Make a video of {description} with leaves gently swaying in the breeze",
            ]
        }
        
        # 保持原有的统一模板列表以兼容现有代码
        self.normal_templates = []
        for templates in self.normal_templates_by_type.values():
            self.normal_templates.extend(templates)
    
    def _load_model(self):
        """加载Qwen2.5-VL模型"""
        try:
            logger.info(f"正在加载Qwen2.5-VL模型: {self.model_name}")
            
            # 使用transformers加载Qwen2.5-VL模型
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            # 确定设备
            if self.device is None or self.device == "cuda":
                device = "cuda"
            elif self.device.startswith("cuda:"):
                device = self.device
            elif self.device == "cpu":
                device = "cpu"
            else:
                print(f"不支持的device参数: {self.device}")
                raise ValueError(f"不支持的device参数: {self.device}")
            
            # 加载模型和处理器
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            
            # 移动到设备
            self.model = self.model.to(device)
            
            logger.info("Qwen2.5-VL模型加载成功")
        except Exception as e:
            logger.error(f"Qwen2.5-VL模型加载失败: {e}")
            raise
    
    def generate_description(self, image: Image.Image, prompt: str = "Describe this image in 10 words and start with 'the image of'.") -> str:
        """
        为单张图片生成描述
        
        Args:
            image: PIL图像对象
            prompt: 生成描述的提示词
            
        Returns:
            str: 图片描述
        """
        # 使用Qwen2.5-VL的方式生成描述
        
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 应用聊天模板
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 生成输出
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        
        # 解码输出
        description = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        
        # 清理描述文本
        if description:
            # 移除首尾空白
            description = description.strip()
            # 移除Qwen模型的终止符
            description = description.replace("<|im_end|>", "").strip()
            # 将描述转换为小写
            description = description.lower()
            # 移除句号等标点符号
            if description.endswith(('.', '!', '?')):
                description = description[:-1]
        
        return description if description else "A picture"
    
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
    
    def get_dataset_type(self, dataset_name: str) -> str:
        """
        根据数据集名称获取类型
        
        Args:
            dataset_name: 数据集名称 (Flickr30k, AFHQ-V2, LHQ)
            
        Returns:
            str: 数据集类型 ('people', 'animals', 'landscape')
        """
        return self.dataset_type_mapping.get(dataset_name, 'landscape')
    
    def generate_normal_prompt(self, description: str, template_idx: Optional[int] = None, dataset_type: Optional[str] = None) -> str:
        """
        基于描述生成正常prompt
        
        Args:
            description: 图片描述
            template_idx: 模板索引，None时随机选择
            dataset_type: 数据集类型，None时使用所有模板
            
        Returns:
            str: 正常prompt
        """
        import random
        
        # 获取对应类型的模板
        if dataset_type and dataset_type in self.normal_templates_by_type:
            templates = self.normal_templates_by_type[dataset_type]
        else:
            # 如果类型不存在，使用所有模板
            templates = self.normal_templates
        
        # 选择模板
        if template_idx is None:
            template_idx = random.randint(0, len(templates) - 1)
        
        template = templates[template_idx % len(templates)]
        return template.format(description=description)
    
    def process_images_with_descriptions(
        self, 
        images: List[Image.Image], 
        save_json_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict:
        """
        处理图像列表，生成描述、恶意prompts和正常prompts

        Args:
            images: PIL图像列表
            save_json_path: JSON保存路径，None时不保存
            dataset_name: 数据集名称 (Flickr30k, AFHQ-V2, LHQ)，None时使用所有模板
        Returns:
            Dict: (包含描述、恶意prompts和正常prompts的字典)
        """
        logger.info(f"开始处理 {len(images)} 张图像")
        
        # 根据数据集名称获取类型
        dataset_type = None
        if dataset_name:
            dataset_type = self.get_dataset_type(dataset_name)
            logger.info(f"使用数据集: {dataset_name} -> 类型: {dataset_type}")

        # 生成描述
        descriptions = []
        for i, image in enumerate(images):
            logger.info(f"正在处理第 {i+1}/{len(images)} 张图片")
            description = self.generate_description(image)
            descriptions.append(description)

        # 生成恶意prompts
        malicious_prompts = []
        for description in descriptions:
            malicious_prompt = self.generate_malicious_prompt(description)
            malicious_prompts.append(malicious_prompt)

        # 生成正常prompts
        normal_prompts = []
        for description in descriptions:
            normal_prompt = self.generate_normal_prompt(description, dataset_type=dataset_type)
            normal_prompts.append(normal_prompt)

        # 准备结果字典
        result_dict = {
            "images_count": len(images),
            "data": []
        }

        for i in range(len(images)):
            item_data = {
                "image_id": i,
                "description": descriptions[i],
                "malicious_prompt": malicious_prompts[i],
                "normal_prompt": normal_prompts[i]
            }
            
            # 如果指定了数据集名称，添加类型信息
            if dataset_name:
                item_data["dataset_name"] = dataset_name
                item_data["dataset_type"] = dataset_type
            
            result_dict["data"].append(item_data)

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
        print("found the file")
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
            num = data.get("images_count", 0)
            print(f"image_count:{num}")
            return data.get("images_count", 0) >= expected_count
    except:
        return False
