import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from train.models.base_model import BaseModel
from train.models.EffoNav.self_attention import MultiLayerDecoder
from torchvision import transforms as T
from einops.einops import rearrange

from train.models.EffoNav.position_encoding import PositionEncodingSine
from train.models.EffoNav.cross_attention.cross_attn import CrossAttn

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class EffoNav(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(EffoNav, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        self.dino_channel=384 #768

        self.num_goal_features=256
        self.goal_encoder_pe = PositionEncodingSine(self.dino_channel,(12,12))
        self.goal_attn = CrossAttn(self.dino_channel,8,['self', 'cross'] * 4)

        block=BasicBlock
        self.in_planes=self.dino_channel

        layer1 = self._make_layer(block, self.dino_channel, stride=1)  
        layer2 = self._make_layer(block, 1024, stride=1)  
        layer3 = self._make_layer(block, self.dino_channel, stride=1)  
        self.backbone=nn.Sequential(layer1,layer2,layer3)

        self.compress_goal_enc_new = nn.Linear(self.dino_channel, self.goal_encoding_size)
        self.goal_conv=nn.Sequential(nn.Conv2d(self.dino_channel*2,self.dino_channel,3,2,1),
                                     nn.BatchNorm2d(self.dino_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(self.dino_channel,self.dino_channel,3,2,1),
                                     nn.BatchNorm2d(self.dino_channel),
                                     nn.ReLU(),)
        
        self.compress_current_obs_enc_new = nn.Linear(self.dino_channel, self.goal_encoding_size)
        self.current_obs_conv=nn.Sequential(nn.Conv2d(self.dino_channel,self.dino_channel,3,2,1),
                                     nn.BatchNorm2d(self.dino_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(self.dino_channel,self.dino_channel,3,2,1),
                                     nn.BatchNorm2d(self.dino_channel),
                                     nn.ReLU(),)

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,DINO=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B,_,h, w=goal_img.shape

        img_0 = T.Resize((168, 168))(obs_img[:, 3*self.context_size:, :, :])
        img_1 = T.Resize((168, 168))(goal_img)

        _,_,h_new,w_new=img_0.shape
        
        ph,pw=h_new//14,w_new//14
        
        with torch.no_grad():
            feats_goal = DINO(torch.cat([img_0,img_1], dim=0))
        feats_goal=feats_goal.permute(0,2,1).reshape(2*B,-1,ph,pw)


        # feats_goal = rearrange(self.goal_encoder_pe(feats_goal), 'n c h w -> n (h w) c')

        (feat_0, feat_1) = feats_goal.split(B)
        dino_channel=feat_0.shape[-1]
        
        feat_0=self.backbone(feat_0) 
        
        current_obs_encoding=feat_0.clone()
         
        feat_1=self.backbone(feat_1)

        feat_0 = rearrange(self.goal_encoder_pe(feat_0), 'n c h w -> n (h w) c')
        feat_1 = rearrange(self.goal_encoder_pe(feat_1), 'n c h w -> n (h w) c')

        feat_0, feat_1 = self.goal_attn(feat_0, feat_1)
        feat_0 = feat_0.permute(0,2,1).reshape(B,-1,ph,pw)
        feat_1 = feat_1.permute(0,2,1).reshape(B,-1,ph,pw)


        goal_encoding=self.goal_conv(torch.cat([feat_0,feat_1],dim=1))
        goal_encoding=nn.AdaptiveAvgPool2d((1,1))(goal_encoding)
        goal_encoding=torch.flatten(goal_encoding,start_dim=1)
        goal_encoding=self.compress_goal_enc_new(goal_encoding)  
        goal_encoding=goal_encoding.unsqueeze(1)

        current_obs_encoding=self.current_obs_conv(current_obs_encoding)
        current_obs_encoding=nn.AdaptiveAvgPool2d((1,1))(current_obs_encoding)
        current_obs_encoding=torch.flatten(current_obs_encoding,start_dim=1)
        current_obs_encoding=self.compress_current_obs_enc_new(current_obs_encoding)  
        current_obs_encoding=current_obs_encoding.unsqueeze(1)
        
        # split the observation into context based on the context size
        # image size is [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img[:-1], dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size), self.obs_encoding_size]
        # reshape the obs_encoding to [context, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, current_obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred