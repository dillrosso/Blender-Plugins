bl_info = {
    "name": "Auto-Heightmap",
    "author": "DillanBrogan",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Terrain Tools",
    "description": "Generate heightmap from selected terrain mesh",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
from mathutils import Vector
import numpy as np
from bpy.props import IntProperty, FloatProperty, StringProperty, BoolProperty
from bpy.types import Panel, Operator
import os

class TERRAIN_OT_create_heightmap(Operator):
    """Create heightmap from selected terrain object"""
    bl_idname = "terrain.create_heightmap"
    bl_label = "Create Heightmap"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Properties
    resolution: IntProperty(
        name="Resolution",
        description="Heightmap resolution (width x height)",
        default=512,
        min=64,
        max=4096
    )
    
    invert_height: BoolProperty(
        name="Invert Height",
        description="Invert height values (black=high, white=low)",
        default=False
    )
    
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of smoothing passes",
        default=0,
        min=0,
        max=10
    )
    
    file_path: StringProperty(
        name="File Path",
        description="Output file path",
        default="//heightmap.png",
        subtype='FILE_PATH'
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.active_object.select_get())
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object")
            return {'CANCELLED'}
        
        try:
            # Create heightmap
            heightmap_data = self.create_heightmap_from_mesh(obj)
            
            # Create image in Blender
            image_name = f"{obj.name}_heightmap"
            self.create_blender_image(heightmap_data, image_name)
            
            # Save to file if path specified
            if self.file_path and self.file_path != "//heightmap.png":
                self.save_heightmap_image(heightmap_data, self.file_path)
            
            self.report({'INFO'}, f"Heightmap created successfully: {image_name}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error creating heightmap: {str(e)}")
            return {'CANCELLED'}
    
    def create_heightmap_from_mesh(self, obj):
        """Generate heightmap data from mesh object"""
        
        # Get mesh data
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        
        # Transform vertices to world coordinates
        vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
        
        if not vertices:
            raise Exception("Mesh has no vertices")
        
        # Find bounds
        min_x = min(v.x for v in vertices)
        max_x = max(v.x for v in vertices)
        min_y = min(v.y for v in vertices)
        max_y = max(v.y for v in vertices)
        min_z = min(v.z for v in vertices)
        max_z = max(v.z for v in vertices)
        
        # Create heightmap array
        heightmap = np.zeros((self.resolution, self.resolution))
        
        # Map vertices to heightmap
        for vertex in vertices:
            # Normalize X,Y coordinates to heightmap space
            if max_x != min_x:
                x_norm = (vertex.x - min_x) / (max_x - min_x)
            else:
                x_norm = 0.5
                
            if max_y != min_y:
                y_norm = (vertex.y - min_y) / (max_y - min_y)
            else:
                y_norm = 0.5
            
            # Convert to pixel coordinates
            px = int(x_norm * (self.resolution - 1))
            py = int(y_norm * (self.resolution - 1))
            
            # Normalize height
            if max_z != min_z:
                height_norm = (vertex.z - min_z) / (max_z - min_z)
            else:
                height_norm = 0.5
            
            # Store maximum height at each pixel (for overlapping points)
            heightmap[py, px] = max(heightmap[py, px], height_norm)
        
        # Fill empty pixels using interpolation
        heightmap = self.fill_empty_pixels(heightmap)
        
        # Apply smoothing if requested
        for _ in range(self.smooth_iterations):
            heightmap = self.smooth_heightmap(heightmap)
        
        # Invert if requested
        if self.invert_height:
            heightmap = 1.0 - heightmap
        
        # Convert to 0-255 range
        heightmap = (heightmap * 255).astype(np.uint8)
        
        # Clean up
        eval_obj.to_mesh_clear()
        
        return heightmap
    
    def fill_empty_pixels(self, heightmap):
        """Fill empty pixels using nearest neighbor interpolation"""
        from scipy import ndimage
        
        # Create mask of non-zero pixels
        mask = heightmap > 0
        
        if not np.any(mask):
            return heightmap
        
        # Find indices of non-zero pixels
        indices = np.where(mask)
        
        # Create coordinate arrays
        coords = np.array(list(zip(indices[0], indices[1])))
        values = heightmap[mask]
        
        # Fill empty pixels
        for i in range(heightmap.shape[0]):
            for j in range(heightmap.shape[1]):
                if heightmap[i, j] == 0:
                    # Find nearest non-zero pixel
                    distances = np.sum((coords - np.array([i, j]))**2, axis=1)
                    nearest_idx = np.argmin(distances)
                    heightmap[i, j] = values[nearest_idx]
        
        return heightmap
    
    def smooth_heightmap(self, heightmap):
        """Apply smoothing filter to heightmap"""
        from scipy import ndimage
        return ndimage.gaussian_filter(heightmap, sigma=1.0)
    
    def create_blender_image(self, heightmap_data, image_name):
        """Create image in Blender from heightmap data"""
        
        # Remove existing image if it exists
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        
        # Create new image
        image = bpy.data.images.new(
            name=image_name,
            width=self.resolution,
            height=self.resolution,
            alpha=False
        )
        
        # Convert heightmap to RGBA
        pixels = []
        for row in heightmap_data:
            for pixel in row:
                normalized = pixel / 255.0
                pixels.extend([normalized, normalized, normalized, 1.0])
        
        # Set pixels
        image.pixels = pixels
        image.pack()
        
        # Update image
        image.update()
    
    def save_heightmap_image(self, heightmap_data, filepath):
        """Save heightmap to file"""
        try:
            from PIL import Image
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(heightmap_data, mode='L')
            
            # Resolve Blender relative path
            if filepath.startswith('//'):
                filepath = bpy.path.abspath(filepath)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save image
            img.save(filepath)
            
        except ImportError:
            # Fallback: use Blender's image save
            image_name = f"temp_heightmap_{self.resolution}"
            self.create_blender_image(heightmap_data, image_name)
            
            image = bpy.data.images[image_name]
            image.filepath_raw = filepath
            image.file_format = 'PNG'
            image.save()
            
            # Clean up temp image
            bpy.data.images.remove(image)

class TERRAIN_PT_heightmap_panel(Panel):
    """Panel for terrain heightmap tools"""
    bl_label = "Terrain to Heightmap"
    bl_idname = "TERRAIN_PT_heightmap"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Terrain Tools"
    
    def draw(self, context):
        layout = self.layout
        
        # Check if valid object is selected
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.select_get():
            layout.label(text=f"Selected: {obj.name}", icon='MESH_DATA')
        else:
            layout.label(text="Select a mesh object", icon='ERROR')
        
        # Operator button
        layout.separator()
        op = layout.operator("terrain.create_heightmap", text="Generate Heightmap", icon='IMAGE_DATA')
        
        # Settings
        layout.separator()
        layout.label(text="Settings:")
        
        scene = context.scene
        if hasattr(scene, 'terrain_heightmap_resolution'):
            layout.prop(scene, 'terrain_heightmap_resolution')
        
        # Quick presets
        layout.separator()
        layout.label(text="Quick Presets:")
        row = layout.row()
        row.operator("terrain.heightmap_preset", text="Low (256)").resolution = 256
        row.operator("terrain.heightmap_preset", text="Med (512)").resolution = 512
        row = layout.row()
        row.operator("terrain.heightmap_preset", text="High (1024)").resolution = 1024
        row.operator("terrain.heightmap_preset", text="Ultra (2048)").resolution = 2048

class TERRAIN_OT_heightmap_preset(Operator):
    """Set heightmap resolution preset"""
    bl_idname = "terrain.heightmap_preset"
    bl_label = "Set Resolution Preset"
    
    resolution: IntProperty(default=512)
    
    def execute(self, context):
        # This would be used with the main operator
        return {'FINISHED'}

# Registration
classes = [
    TERRAIN_OT_create_heightmap,
    TERRAIN_OT_heightmap_preset,
    TERRAIN_PT_heightmap_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Add scene properties
    bpy.types.Scene.terrain_heightmap_resolution = IntProperty(
        name="Resolution",
        default=512,
        min=64,
        max=4096
    )

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    # Remove scene properties
    if hasattr(bpy.types.Scene, 'terrain_heightmap_resolution'):
        del bpy.types.Scene.terrain_heightmap_resolution

if __name__ == "__main__":
    register()
