
import torch
from gsplat.exporter import load_ply_to_splats
from gsplat.scene import GaussianScene

import torch.nn.functional as F
import math

import gsplat.geometry.functional.quaternion as quat




class GsplatCombinedObject:
    """
    object in the GaussianScene that can be moved and rotated independently of the scene.
    """

    def __init__(self, name: str, ply_file: str, scene: GaussianScene):
        self.scene=scene
        self.name = name
        self.ply_file = ply_file
        self.splats = load_ply_to_splats(ply_file)
        self.position = torch.tensor([0.0, 0.0, 0.0], device=self.scene.splats["means"].device, dtype=torch.float64)
        self.origin = torch.tensor([0.0, 0.0, 0.0], device=self.scene.splats["means"].device, dtype=torch.float64)
        self.xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.scene.splats["means"].device, dtype=torch.float64)
        self.xyzw_origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.scene.splats["means"].device, dtype=torch.float64)
        self.scale = torch.tensor([1.0, 1.0, 1.0], device=self.scene.splats["means"].device, dtype=torch.float64)

        self._add_to_scene()

    def _add_to_scene(self):
        """
        Add the object to the scene.
        """

        splats_dict = self.splats
        device = self.scene.splats["means"].device
        means = splats_dict["means"].to(device)
        quats = F.normalize(splats_dict["quats"].to(device), p=2, dim=-1)
        scales_raw = splats_dict["scales"].to(device)
        opacities_raw = splats_dict["opacities"].to(device)
        sh0_t = splats_dict["sh0"].to(device)
        shN_t = splats_dict["shN"].to(device)
        colors = torch.cat([sh0_t, shN_t], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means, requires_grad=False),
                "quats": torch.nn.Parameter(quats, requires_grad=False),
                "scales": torch.nn.Parameter(scales_raw, requires_grad=False),
                "opacities": torch.nn.Parameter(opacities_raw, requires_grad=False),
                "colors": torch.nn.Parameter(colors, requires_grad=False),
            }
        )
        self.scene.put(self.name, splats)

    def move_to(self, position: list[float], wxyz: list[float]):
        """
        Move the object in the scene.
        """
        position_t = torch.tensor(position, device=self.scene.splats["means"].device, dtype=torch.float64)
        xyzw_t = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], device=self.scene.splats["means"].device, dtype=torch.float64)  # convert xyzw to wxyz
        old_position = self.position
        old_xyzw = self.xyzw
        self.position = position_t - self.origin
        self.xyzw = quat.quat_multiply(xyzw_t.unsqueeze(0), quat.quat_inverse(self.xyzw_origin.unsqueeze(0))).squeeze(0)
        
        # Calculate the transformation needed
        todo_xyzw = quat.quat_multiply(self.xyzw.unsqueeze(0), quat.quat_inverse(old_xyzw.unsqueeze(0)))
        todo_position = self.position - old_position

        #adjust for origin and position
        origin_adjust = quat.quat_rotate_vector(todo_xyzw, self.origin.unsqueeze(0))
        todo_position_origin_adjust = todo_position - origin_adjust.squeeze(0) + self.origin.unsqueeze(0)
        
        offset_adjust = quat.quat_rotate_vector(todo_xyzw, self.position.unsqueeze(0))
        todo_position_offset_adjust = todo_position_origin_adjust - offset_adjust.squeeze(0) + self.position.unsqueeze(0)
        
        self.scene.transform(self.name, trans=todo_position_offset_adjust, quart=todo_xyzw.squeeze(0))

    def adjust_scale(self, scale: list[float]):
        """
        Adjust the scale of the object in the scene.
        """
        old_scale = self.scale
        cursor_position = self.position + self.origin
        scale_t = torch.tensor(scale, device=self.scene.splats["means"].device, dtype=torch.float64)
        self.scale = scale_t
        todo_scale = scale_t / old_scale
        position_adjust = cursor_position * (1-todo_scale)
        self.origin = self.origin * todo_scale
        self.position = cursor_position - self.origin
        self.scene.transform(self.name, scale=todo_scale, trans=position_adjust)
