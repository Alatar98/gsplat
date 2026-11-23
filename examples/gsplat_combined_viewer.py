# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import viser
import torch
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable
from nerfview import Viewer, RenderTabState

from gsplat.exporter import load_ply_to_splats
from gsplat.rendering import RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.scene import GaussianScene

import torch.nn.functional as F
import math

import gsplat.geometry.functional.quaternion as quat

import tempfile

from gsplat_combined_object import GsplatCombinedObject



class GsplatCombinedObject_viser(GsplatCombinedObject):
    """
    object in the GUI.
    """

    def __init__(self, name: str, ply_file: str, parent):
        self.parent=parent
        self.enabled = True
        self.movable = True
        self.path = None
        self.folder_handle = None

        super().__init__(name, ply_file, parent.scene)

        self.parent.rerender(None)


    def move_to(self, position: list[float], wxyz: list[float]):
        """
        Move the object in the scene. Then rerender the scene.
        """
        super().move_to(position, wxyz)

        self.parent.rerender(None)
        

    def populate_gui(self, server: viser.ViserServer, folder_handle: viser.GuiFolderHandle):
        """
        Populate the GUI for this object.
        """
        if self.folder_handle is not None:
            raise RuntimeError("GUI already populated for this object.")
        self.folder_handle = folder_handle
        with folder_handle:

            divider = server.gui.add_divider()
            show_button = server.gui.add_checkbox(
                f"Show",
                initial_value=self.enabled,
                hint=f"Enable or disable the object {self.name}.",
            )
            @show_button.on_update
            def _(_) -> None:
                self.enabled = show_button.value
                print(f"Object {self.name} enabled: {self.enabled}")

            move_button = server.gui.add_checkbox(
                f"Move",
                initial_value=True,
                hint=f"Enable or disable the object {self.name}.",
            )
            @move_button.on_update
            def _(_) -> None:
                if move_button.value:
                    set_origin_button.value = False
                mover.visible = move_button.value or set_origin_button.value


            set_origin_button = server.gui.add_checkbox(
                "Move Origin",
                initial_value=False,
                hint=f"Move the origin of the object {self.name}.",
            )
            @set_origin_button.on_update
            def _(_) -> None:
                if set_origin_button.value:
                    move_button.value = False
                mover.visible = move_button.value or set_origin_button.value

            mover = server.scene.add_transform_controls(
                name=f"{self.name}/mover",
                position=self.position,
                visible=move_button.value,
            )
            @mover.on_update
            def _(_) -> None:
                if move_button.value:
                    self.move_to(mover.position, mover.wxyz)
                else:
                    self.origin = torch.tensor(mover.position, device=self.scene.splats["means"].device,dtype=torch.float64) - self.position
                    mover_xyzw = torch.tensor([mover.wxyz[1], mover.wxyz[2], mover.wxyz[3], mover.wxyz[0]], device=self.scene.splats["means"].device,dtype=torch.float64)  # convert xyzw to wxyz
                    mover_xyzw = mover_xyzw.reshape(1, 4)
                    self.xyzw_origin = quat.quat_multiply(mover_xyzw, quat.quat_inverse(self.xyzw.unsqueeze(0))).squeeze(0) # convert xyzw to wxyz
                




            #scale

            add_keyframe_button = server.gui.add_button(
                "Add Keyframe",
                icon=viser.Icon.PLUS,
                hint=f"Add a keyframe for object {self.name}.",
            )
            @add_keyframe_button.on_click
            def _(_) -> None:
                # Add a keyframe for the object in the scene.
                print(f"Add a keyframe for the object {self.name}.")


            clear_keyframes_button = server.gui.add_button(
                "Clear Keyframes",
                hint=f"Clear all keyframes for object {self.name}.",
                icon=viser.Icon.TRASH,
            )
            @clear_keyframes_button.on_click
            def _(_) -> None:
                # Clear all keyframes for the object in the scene.
                print(f"Clear all keyframes for the object {self.name}.")

            load_keyframes_button = server.gui.add_upload_button(
                "Load Keyframes",
                icon=viser.Icon.UPLOAD,
                hint=f"Load keyframes for object {self.name}.",
            )
            @load_keyframes_button.on_upload
            def _(_) -> None:
                # Load keyframes for the object in the scene.
                print(f"Load keyframes for the object {self.name}.")

            save_keyframes_button = server.gui.add_button(
                "Save Keyframes",
                icon=viser.Icon.FILE_EXPORT,
                hint=f"Save keyframes for object {self.name}.",
            )
            @save_keyframes_button.on_click
            def _(_) -> None:
                # Save keyframes for the object in the scene.
                print(f"Save keyframes for the object {self.name}.")


            remove_button = server.gui.add_button(
                "Remove",
                icon=viser.Icon.TRASH,
                hint=f"Remove object {self.name} from the scene.",
            )
            @remove_button.on_click
            def _(_) -> None:
                if self.ply_file is not None:
                    if self.ply_file.startswith(tempfile.gettempdir()):
                        try:
                            Path(self.ply_file).unlink()
                        except Exception as e:
                            print(f"Failed to delete temporary file {self.ply_file}: {e}")
                if self.folder_handle is not None:
                    self.folder_handle.remove()
                if self.scene.get(self.name) is not None:
                    self.scene.remove_component(self.name)
                server.scene.remove_by_name(self.name)
                self.parent.rerender(None)




class GsplatCombinedRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0
    near_plane: float = 1e-2
    far_plane: float = 1e2

    # controlable parameters
    max_sh_degree: int = 5
   
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    rasterize_mode: RasterizeMode = "classic"
    camera_model: CameraModel = "pinhole"

class GsplatCombinedViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        scene: GaussianScene,
        mode: Literal["rendering", "training"] = "rendering",
        render_modes: tuple = ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
    ):
        if len(render_modes) == 0:
            raise ValueError("render_modes must contain at least one mode")
        self._render_modes = render_modes
        self.scene = scene
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatCombinedRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            total_gs_count_number = server.gui.add_number(
                "Total",
                initial_value=self.render_tab_state.total_gs_count,
                disabled=True,
                hint="Total number of splats in the scene.",
            )
            rendered_gs_count_number = server.gui.add_number(
                "Rendered",
                initial_value=self.render_tab_state.rendered_gs_count,
                disabled=True,
                hint="Number of splats rendered.",
            )

            max_sh_degree_number = server.gui.add_number(
                "Max SH",
                initial_value=self.render_tab_state.max_sh_degree,
                min=0,
                max=5,
                step=1,
                hint="Maximum SH degree used",
            )

            @max_sh_degree_number.on_update
            def _(_) -> None:
                self.render_tab_state.max_sh_degree = int(
                    max_sh_degree_number.value
                )
                self.rerender(_)

            

            backgrounds_slider = server.gui.add_rgb(
                "Background",
                initial_value=self.render_tab_state.backgrounds,
                hint="Background color for rendering.",
            )

            @backgrounds_slider.on_update
            def _(_) -> None:
                self.render_tab_state.backgrounds = backgrounds_slider.value
                self.rerender(_)


            rasterize_mode_dropdown = server.gui.add_dropdown(
                "Anti-Aliasing",
                ("classic", "antialiased"),
                initial_value=self.render_tab_state.rasterize_mode,
                hint="Whether to use classic or antialiased rasterization.",
            )

            @rasterize_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                self.rerender(_)

            camera_model_dropdown = server.gui.add_dropdown(
                "Camera",
                ("pinhole", "ortho", "fisheye"),
                initial_value=self.render_tab_state.camera_model,
                hint="Camera model used for rendering.",
            )

            @camera_model_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.camera_model = camera_model_dropdown.value
                self.rerender(_)

            self._objects_folder = server.gui.add_folder("Objects")
            with self._objects_folder:
                object_name_field = server.gui.add_text(
                    "Object Name",
                    initial_value="object_0",
                    hint="Name of the object to add.",
                )

                object_file_field = server.gui.add_text(
                    "File Name",
                    initial_value="",
                    hint="Name of the current uploaded object.",
                    disabled=True,
                )


                object_file_button = server.gui.add_upload_button(
                    "Object File",
                    icon=viser.Icon.UPLOAD,
                    hint="Path to the object file.",
                    mime_type=[".ply"],
                )

                @object_file_button.on_upload
                def _(_) -> None:
                    add_object_button.disabled = True
                    object_file_button.disabled = True
                    object_file_field.value = object_file_button.value.name
                    add_object_button.disabled = False
                    object_file_button.disabled = False

                add_object_button = server.gui.add_button(
                    "Add Object",
                    hint="Add an object to the scene.",
                    disabled=True,
                )
                @add_object_button.on_click
                def _(_) -> None:
                    add_object_button.disabled = True
                    object_file_button.disabled = True
                    #check for valid name
                    if not object_name_field.value:
                        print("Object name cannot be empty.")
                        add_object_button.disabled = False
                        object_file_button.disabled = False
                        return
                    for key in self._objects_folder._children.keys():
                        if self._objects_folder._children[key].label == object_name_field.value:
                            print(f"Object name {object_name_field.value} already exists.")
                            add_object_button.disabled = False
                            object_file_button.disabled = False
                            return
                    with self._objects_folder:
                        # Add a new object to the scene
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as f:
                            f.write(object_file_button.value.content)
                            f.flush()
                            temp_file_path = f.name
                        new_object_name = object_name_field.value
                        new_object_state = GsplatCombinedObject_viser(new_object_name, temp_file_path, self)
                        obj_folder = server.gui.add_folder(new_object_name)
                        new_object_state.populate_gui(server, obj_folder)
                    add_object_button.disabled = False
                    object_file_button.disabled = False


        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "backgrounds_slider": backgrounds_slider,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count
