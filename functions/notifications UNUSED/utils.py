"""
Utility functions for notification system
"""

from typing import List, Tuple
from .models import DetectionPoint

def is_point_in_region(point: DetectionPoint, region_data: dict) -> bool:
    """
    Check if a detection point is within any defined region.
    
    This function handles the coordinate system mismatch between:
    - Detection points: normalized (0-1) coordinates from AI inference
    - Region polygons: pixel coordinates from Flutter widget canvas
    
    Args:
        point: DetectionPoint with normalized coordinates (0-1)
        region_data: RegionSelectorData as dict with polygon coordinates
        
    Returns:
        True if point is within any defined region
    """
    try:
        # DEBUG: Log input data
        print(f"DEBUG: Checking point ({point.x}, {point.y}) against region")
        
        paths_data = region_data.get("paths", [])
        print(f"DEBUG: Region has {len(paths_data)} paths")
        
        if not paths_data:
            print("DEBUG: No paths in region data")
            return False
        
        # STEP 1: Infer canvas size from polygon coordinates
        # This is the key fix - we determine the actual canvas size from the polygon data
        all_x_coords = []
        all_y_coords = []
        
        for path_info in paths_data:
            points_data = path_info.get("points", [])
            for p in points_data:
                all_x_coords.append(p.get("x", 0))
                all_y_coords.append(p.get("y", 0))
        
        if not all_x_coords or not all_y_coords:
            print("DEBUG: No valid polygon points found")
            return False
        
        # Calculate the bounding box of all polygon points
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        
        # Infer canvas dimensions with some padding
        # The polygon extends to the edges of the canvas, so we add some margin
        canvas_width = max(max_x - min_x, max_x + abs(min_x)) + 20
        canvas_height = max(max_y - min_y, max_y + abs(min_y)) + 20
        
        print(f"DEBUG: Inferred canvas size: {canvas_width}x{canvas_height}")
        print(f"DEBUG: Polygon bounds: X({min_x}, {max_x}), Y({min_y}, {max_y})")
        
        # STEP 2: Convert normalized detection point to pixel coordinates
        # using the inferred canvas size
        pixel_x = point.x * canvas_width
        pixel_y = point.y * canvas_height
        
        print(f"DEBUG: Converted point to pixels: ({pixel_x}, {pixel_y})")
        
        # STEP 3: Check each polygon path
        for path_idx, path_info in enumerate(paths_data):
            operation = path_info.get("operation", "add")
            points_data = path_info.get("points", [])
            
            print(f"DEBUG: Path {path_idx}: operation={operation}, points={len(points_data)}")
            
            if not points_data:
                continue
                
            # Create polygon from points
            polygon_points = []
            for p in points_data:
                px, py = p.get("x", 0), p.get("y", 0)
                polygon_points.append((px, py))
                
            print(f"DEBUG: Polygon points: {polygon_points[:3]}...")  # Show first 3 points
            
            # Check if point is inside polygon using ray casting
            if _point_in_polygon(pixel_x, pixel_y, polygon_points):
                print(f"DEBUG: Point is inside polygon {path_idx}")
                
                # Return True if operation is "add" (inclusion zone)
                # Return False if operation is "remove" (exclusion zone)
                return operation == "add"
                
        print("DEBUG: Point not in any polygon")
        return False
        
    except Exception as e:
        print(f"DEBUG: Error in is_point_in_region: {e}")
        return False

def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm to determine if a point is inside a polygon.
    
    Args:
        x, y: Point coordinates
        polygon: List of (x, y) tuples defining polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside