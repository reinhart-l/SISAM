from sam_unet.models.sam_unet_model import SAM_UNET
from sam_unet.models.segment_anything.build_sam import sam_model_registry
from sam_unet.config import config_dict
import torch


def build_res50_sam(need_ori_checkpoint, sam_unet_checkpoint):
    return _build_sam(
        type=50,
        ori_sam=sam_model_registry['vit_b_1024'],
        ori_checkpoint=need_ori_checkpoint,
        sam_unet_checkpoint=sam_unet_checkpoint,
    )


def build_res34_sam(need_ori_checkpoint, sam_unet_checkpoint):
    return _build_sam(
        type=34,
        ori_sam=sam_model_registry['vit_b_1024'],
        ori_checkpoint=need_ori_checkpoint,
        sam_unet_checkpoint=sam_unet_checkpoint,
    )

sam_unet_registry = {
    'res50_sam_unet': build_res50_sam,
    'res34_sam_unet': build_res34_sam,
}


def _build_sam(type, 
               ori_sam, 
               ori_checkpoint: bool, 
               sam_unet_checkpoint: str):
    if sam_unet_checkpoint is not None and ori_checkpoint == False:
        sam_unet_checkpoint = torch.load(sam_unet_checkpoint)
        total_params = 0
        # for key, value in sam_unet_checkpoint['model'].items():
        #     # 确保 value 是张量
        #     if isinstance(value, torch.Tensor):
        #         total_params += value.numel()  # 计算每个参数的数量
        # 
        # # 计算理论显存占用（假设使用 float32）
        # memory_usage_MB = total_params * 4 / (1024 ** 2)  # 转换为 MB
        # 
        # print(f"Total parameters: {total_params}")
        # print(f"Theoretical memory usage: {memory_usage_MB:.2f} MB")
        # print("Keys in sam_unet_checkpoint:")
        # for key in sam_unet_checkpoint['model'].keys():
        #     print(key)
        new_sam_unet_checkpoint = {k.replace('module.', ''): v for k, v in sam_unet_checkpoint['model'].items()}

        if 'model' in new_sam_unet_checkpoint.keys():
            new_sam_unet_checkpoint = new_sam_unet_checkpoint['model']
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam(None), is_resnet_pretrained_or_not=False)

        model.load_state_dict(new_sam_unet_checkpoint)

    elif ori_checkpoint == True: 
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam(config_dict['checkpoint_path']), is_resnet_pretrained_or_not=True)
    else:
        model = SAM_UNET(resnet_type=type, ori_sam=ori_sam(None), is_resnet_pretrained_or_not=False)
    return model
