#!/usr/bin/env python3
"""
Gaussian Splat Bounds Processor
Processes PLY or SPZ Gaussian splat files to detect outliers and calculate precise bounds
Outputs Cesium-compatible clipping polygon coordinates
"""

import numpy as np
import argparse
import json
import gzip
import struct
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SplatPoint:
    """Represents a single Gaussian splat point with position and properties"""
    x: float
    y: float
    z: float
    opacity: float = 1.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

class GaussianSplatProcessor:
    """Main processor for Gaussian splat files"""
    
    def __init__(self, file_path: str, coordinate_transform: str = 'none', scale_factor: float = 1.0):
        self.file_path = file_path
        self.coordinate_transform = coordinate_transform
        self.scale_factor = scale_factor
        self.points: List[SplatPoint] = []
        self.file_type = self._detect_file_type()
        
    def _detect_file_type(self) -> str:
        """Detect file type based on extension"""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.ply':
            return 'ply'
        elif ext == '.spz':
            return 'spz'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def load_points(self) -> None:
        """Load points from the splat file"""
        if self.file_type == 'ply':
            self._load_ply()
        elif self.file_type == 'spz':
            self._load_spz()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        print(f"Loaded {len(self.points)} points from {self.file_path}")
    
    def _load_ply(self) -> None:
        """Load points from PLY file"""
        # Try UTF-8 first, then fall back to binary mode for mixed files
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return self._read_ply_file(f)
        except UnicodeDecodeError:
            # Handle binary PLY files or files with mixed encoding
            return self._load_ply_binary_mode()
    
    def _read_ply_file(self, f) -> None:
        """Read PLY file content"""
        # Read header
        line = f.readline().strip()
        if not line.startswith('ply'):
            raise ValueError("Not a valid PLY file")
        
        # Parse header
        vertex_count = 0
        properties = []
        is_binary = False
        
        while True:
            line = f.readline().strip()
            if line == 'end_header':
                break
            elif line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
        
        # Read data based on format
        if is_binary:
            raise ValueError("Binary PLY detected. Please convert to ASCII format or use _load_ply_binary_mode()")
        else:
            self._read_ply_ascii(f, vertex_count, properties)
    
    def _load_ply_binary_mode(self) -> None:
        """Load PLY file in binary mode to handle mixed encoding"""
        print("Attempting to read PLY file in binary mode...")
        
        with open(self.file_path, 'rb') as f:
            # Read header in binary mode
            header_lines = []
            current_line = b''
            
            while True:
                byte = f.read(1)
                if not byte:
                    break
                
                if byte == b'\n':
                    line = current_line.decode('utf-8', errors='ignore').strip()
                    header_lines.append(line)
                    
                    if line == 'end_header':
                        break
                    current_line = b''
                else:
                    current_line += byte
            
            # Parse header
            vertex_count = 0
            properties = []
            is_binary = False
            
            for line in header_lines:
                if line.startswith('format'):
                    if 'binary' in line:
                        is_binary = True
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    if len(parts) >= 3:
                        prop_type = parts[1]
                        prop_name = parts[2]
                        properties.append((prop_name, prop_type))
            
            print(f"Header parsed: {vertex_count} vertices, binary={is_binary}")
            
            if is_binary:
                self._read_ply_binary_data(f, vertex_count, properties)
            else:
                # Read ASCII data in binary mode
                self._read_ply_ascii_from_binary(f, vertex_count, properties)
    
    def _read_ply_ascii(self, f, vertex_count: int, properties: List[Tuple[str, str]]) -> None:
        """Read ASCII PLY data"""
        prop_names = [prop[0] for prop in properties]
        
        for _ in range(vertex_count):
            line = f.readline().strip()
            if not line:
                break
            
            values = line.split()
            point_data = dict(zip(prop_names, values))
            
            # Extract position (required)
            x = float(point_data.get('x', 0))
            y = float(point_data.get('y', 0))
            z = float(point_data.get('z', 0))
            
            # Extract optional properties
            opacity = float(point_data.get('opacity', 1.0))
            scale_x = float(point_data.get('scale_0', 1.0))
            scale_y = float(point_data.get('scale_1', 1.0))
            scale_z = float(point_data.get('scale_2', 1.0))
            
            # Apply coordinate transformation if specified
            x_transformed, y_transformed, z_transformed = self._apply_coordinate_transform(x, y, z)
            
            # Apply scaling if specified
            x_final, y_final, z_final = self._apply_scale_factor(x_transformed, y_transformed, z_transformed)
            
            self.points.append(SplatPoint(x_final, y_final, z_final, opacity, scale_x, scale_y, scale_z))
    
    def _read_ply_ascii_from_binary(self, f, vertex_count: int, properties: List[Tuple[str, str]]) -> None:
        """Read ASCII PLY data from binary file handle"""
        prop_names = [prop[0] for prop in properties]
        
        points_read = 0
        while points_read < vertex_count:
            # Read line in binary mode
            line_bytes = b''
            while True:
                byte = f.read(1)
                if not byte or byte == b'\n':
                    break
                line_bytes += byte
            
            if not line_bytes:
                break
                
            try:
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                values = line.split()
                if len(values) < len(prop_names):
                    continue
                    
                point_data = dict(zip(prop_names, values))
                
                # Extract position (required)
                x = float(point_data.get('x', 0))
                y = float(point_data.get('y', 0))
                z = float(point_data.get('z', 0))
                
                # Extract optional properties
                opacity = float(point_data.get('opacity', 1.0))
                scale_x = float(point_data.get('scale_0', 1.0))
                scale_y = float(point_data.get('scale_1', 1.0))
                scale_z = float(point_data.get('scale_2', 1.0))
                
                # Apply coordinate transformation if specified
                x_transformed, y_transformed, z_transformed = self._apply_coordinate_transform(x, y, z)
                
                # Apply scaling if specified
                x_final, y_final, z_final = self._apply_scale_factor(x_transformed, y_transformed, z_transformed)
                
                self.points.append(SplatPoint(x_final, y_final, z_final, opacity, scale_x, scale_y, scale_z))
                points_read += 1
                
            except (ValueError, UnicodeDecodeError) as e:
                print(f"Skipping malformed line {points_read}: {e}")
                continue
    
    def _read_ply_binary_data(self, f, vertex_count: int, properties: List[Tuple[str, str]]) -> None:
        """Read binary PLY data"""
        print("Reading binary PLY data...")
        
        # Create property map for binary reading
        prop_formats = []
        prop_names = []
        
        for prop_name, prop_type in properties:
            prop_names.append(prop_name)
            if prop_type == 'float':
                prop_formats.append('f')
            elif prop_type == 'double':
                prop_formats.append('d')
            elif prop_type in ['uchar', 'uint8']:
                prop_formats.append('B')
            elif prop_type in ['char', 'int8']:
                prop_formats.append('b')
            elif prop_type in ['ushort', 'uint16']:
                prop_formats.append('H')
            elif prop_type in ['short', 'int16']:
                prop_formats.append('h')
            elif prop_type in ['uint', 'uint32']:
                prop_formats.append('I')
            elif prop_type in ['int', 'int32']:
                prop_formats.append('i')
            else:
                prop_formats.append('f')  # Default to float
        
        # Calculate bytes per vertex
        format_string = '<' + ''.join(prop_formats)  # Little endian
        bytes_per_vertex = struct.calcsize(format_string)
        
        print(f"Binary format: {format_string}, {bytes_per_vertex} bytes per vertex")
        
        for i in range(vertex_count):
            try:
                vertex_data = f.read(bytes_per_vertex)
                if len(vertex_data) < bytes_per_vertex:
                    break
                
                values = struct.unpack(format_string, vertex_data)
                point_data = dict(zip(prop_names, values))
                
                # Extract position (required)
                x = float(point_data.get('x', 0))
                y = float(point_data.get('y', 0))
                z = float(point_data.get('z', 0))
                
                # Extract optional properties
                opacity = float(point_data.get('opacity', 1.0))
                scale_x = float(point_data.get('scale_0', 1.0))
                scale_y = float(point_data.get('scale_1', 1.0))
                scale_z = float(point_data.get('scale_2', 1.0))
                
                # Apply coordinate transformation if specified
                x_transformed, y_transformed, z_transformed = self._apply_coordinate_transform(x, y, z)
                
                # Apply scaling if specified
                x_final, y_final, z_final = self._apply_scale_factor(x_transformed, y_transformed, z_transformed)
                
                self.points.append(SplatPoint(x_final, y_final, z_final, opacity, scale_x, scale_y, scale_z))
                
            except struct.error as e:
                print(f"Error reading vertex {i}: {e}")
                break
    
    def _load_spz(self) -> None:
        """Load points from SPZ file using proper format specification"""
        try:
            with gzip.open(self.file_path, 'rb') as f:
                # Read SPZ header (16 bytes)
                header = f.read(16)
                if len(header) < 16:
                    raise ValueError("Invalid SPZ file: header too short")
                
                # Parse header according to SPZ specification
                magic = struct.unpack('<I', header[0:4])[0]
                version = struct.unpack('<I', header[4:8])[0]
                point_count = struct.unpack('<I', header[8:12])[0]
                sh_degree = header[12]
                fractional_bits = header[13]
                flags = header[14]
                reserved = header[15]
                
                # Validate magic number
                if magic != 0x5053474e:  # "PSGN" in little endian
                    raise ValueError(f"Invalid SPZ magic number: {magic:08x} (expected 0x5053474e)")
                
                # Currently only support version 2
                if version != 2:
                    raise ValueError(f"Unsupported SPZ version: {version} (only version 2 supported)")
                
                if reserved != 0:
                    print(f"Warning: Reserved field should be 0, got {reserved}")
                
                print(f"SPZ file: version {version}, {point_count} points, SH degree {sh_degree}, fractional bits {fractional_bits}")
                
                # Calculate spherical harmonics coefficients count
                sh_coeffs_count = (sh_degree + 1) ** 2
                
                # SPZ v2 data layout: all positions first, then all other attributes
                # Position data: 3 * 24-bit signed integers per point = 9 bytes per point
                positions_size = point_count * 9
                
                # Read all position data
                positions_data = f.read(positions_size)
                if len(positions_data) < positions_size:
                    raise ValueError(f"Incomplete position data: expected {positions_size} bytes, got {len(positions_data)}")
                
                # Parse positions from 24-bit fixed-point integers
                points_loaded = 0
                points_skipped = 0
                max_points = point_count  # Process all points (removed artificial limit)
                
                print(f"Parsing {max_points} positions from 24-bit fixed-point data...")
                
                for i in range(max_points):
                    try:
                        # Each position is 9 bytes: 3 x 24-bit signed integers
                        pos_offset = i * 9
                        if pos_offset + 9 > len(positions_data):
                            break
                        
                        # Extract 24-bit signed integers (3 bytes each)
                        x_bytes = positions_data[pos_offset:pos_offset+3]
                        y_bytes = positions_data[pos_offset+3:pos_offset+6]
                        z_bytes = positions_data[pos_offset+6:pos_offset+9]
                        
                        # Convert 24-bit signed integers to float
                        x_int = self._bytes_to_24bit_signed(x_bytes)
                        y_int = self._bytes_to_24bit_signed(y_bytes)
                        z_int = self._bytes_to_24bit_signed(z_bytes)
                        
                        # Convert fixed-point to float using fractional_bits
                        scale_factor = 1.0 / (1 << fractional_bits)
                        x = x_int * scale_factor
                        y = y_int * scale_factor
                        z = z_int * scale_factor
                        
                        # Debug: Check for potential integer overflow issues
                        if i < 5:  # Show first 5 raw values for debugging
                            print(f"  Raw point {i}: int=({x_int}, {y_int}, {z_int}), float=({x:.2f}, {y:.2f}, {z:.2f})")
                        
                        # Validate coordinates
                        if (np.isnan(x) or np.isnan(y) or np.isnan(z) or 
                            np.isinf(x) or np.isinf(y) or np.isinf(z)):
                            points_skipped += 1
                            continue
                        
                        # Apply coordinate transformation if specified
                        x_transformed, y_transformed, z_transformed = self._apply_coordinate_transform(x, y, z)
                        
                        # Apply scaling if specified
                        x_final, y_final, z_final = self._apply_scale_factor(x_transformed, y_transformed, z_transformed)
                        
                        # Debug: Show transformation and scaling effect on first few points
                        if i < 5 and (self.coordinate_transform != 'none' or self.scale_factor != 1.0):
                            if self.scale_factor != 1.0:
                                print(f"  Transform {self.coordinate_transform} + scale {self.scale_factor}: ({x:.2f}, {y:.2f}, {z:.2f}) → ({x_final:.2f}, {y_final:.2f}, {z_final:.2f})")
                            else:
                                print(f"  Transform {self.coordinate_transform}: ({x:.2f}, {y:.2f}, {z:.2f}) → ({x_final:.2f}, {y_final:.2f}, {z_final:.2f})")
                        
                        self.points.append(SplatPoint(x_final, y_final, z_final, 1.0))
                        points_loaded += 1
                        
                    except Exception as e:
                        points_skipped += 1
                        if points_skipped < 10:  # Only show first few errors
                            print(f"Skipping malformed point {i}: {e}")
                        continue
                
                print(f"SPZ loading complete: {points_loaded} points loaded, {points_skipped} points skipped")
                
                if points_loaded == 0:
                    raise ValueError("No valid points could be loaded from SPZ file")
                
                # Debug: Analyze coordinate ranges
                if self.points:
                    x_coords = [p.x for p in self.points]
                    y_coords = [p.y for p in self.points]
                    z_coords = [p.z for p in self.points]
                    
                    print(f"SPZ coordinate ranges:")
                    print(f"  X: {min(x_coords):.2f} to {max(x_coords):.2f} (range: {max(x_coords)-min(x_coords):.2f})")
                    print(f"  Y: {min(y_coords):.2f} to {max(y_coords):.2f} (range: {max(y_coords)-min(y_coords):.2f})")
                    print(f"  Z: {min(z_coords):.2f} to {max(z_coords):.2f} (range: {max(z_coords)-min(z_coords):.2f})")
                    
                    # Analyze coordinate distribution
                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    z_center = sum(z_coords) / len(z_coords)
                    print(f"Coordinate centers: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f})")
                    
                    # Check if most points are near center vs edges
                    center_radius = 500  # Points within 500 units of center
                    near_center = sum(1 for p in self.points if abs(p.x - x_center) < center_radius and abs(p.y - y_center) < center_radius and abs(p.z - z_center) < center_radius)
                    print(f"Points within {center_radius} units of center: {near_center}/{len(self.points)} ({near_center/len(self.points)*100:.1f}%)")
                    
                    # Show first few points for debugging
                    print(f"First 5 points:")
                    for i, p in enumerate(self.points[:5]):
                        print(f"  Point {i}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
                    
        except Exception as e:
            print(f"Error reading SPZ file: {e}")
            print("If this persists, try using the original PLY file instead.")
            raise
    
    def _bytes_to_24bit_signed(self, bytes_data: bytes) -> int:
        """Convert 3 bytes to 24-bit signed integer"""
        if len(bytes_data) != 3:
            raise ValueError(f"Expected 3 bytes, got {len(bytes_data)}")
        
        # Convert 3 bytes to unsigned 24-bit integer (little endian)
        unsigned_val = bytes_data[0] | (bytes_data[1] << 8) | (bytes_data[2] << 16)
        
        # Convert to signed 24-bit integer
        if unsigned_val >= (1 << 23):  # If sign bit is set
            return unsigned_val - (1 << 24)
        else:
            return unsigned_val
    
    def _apply_coordinate_transform(self, x: float, y: float, z: float) -> tuple:
        """Apply coordinate transformation based on the specified transform type"""
        if self.coordinate_transform == 'none':
            return x, y, z
        elif self.coordinate_transform == 'y_up_to_z_up':
            # Convert Y-up coordinate system to Z-up (common for graphics software)
            return x, z, -y
        elif self.coordinate_transform == 'swap_yz':
            # Simple Y-Z swap
            return x, z, y
        elif self.coordinate_transform == 'rdf_to_luf':
            # Convert RDF (Right-Down-Front) to LUF (Left-Up-Front) coordinate system
            # This matches the transformation applied during PLY→SPZ conversion
            return -x, -y, z
        elif self.coordinate_transform == 'luf_to_rdf':
            # Convert LUF (Left-Up-Front) back to RDF (Right-Down-Front) coordinate system
            # Use this for SPZ files that were converted from PLY and need to match Cesium expectations
            return -x, -y, z
        elif self.coordinate_transform == 'rub_to_rdf':
            # Convert RUB (Right-Up-Back, OpenGL/three.js) to RDF (Right-Down-Front, Cesium)
            # SPZ uses RUB coordinate system internally
            return x, -y, -z
        elif self.coordinate_transform == 'flip_y':
            # Simple Y-axis flip
            return x, -y, z
        elif self.coordinate_transform == 'flip_z':
            # Simple Z-axis flip  
            return x, y, -z
        elif self.coordinate_transform == 'flip_yz':
            # Flip both Y and Z axes
            return x, -y, -z
        elif self.coordinate_transform == 'y_up_to_z_up_flip_x':
            # Y-up to Z-up conversion + flip X axis
            return -x, z, -y
        elif self.coordinate_transform == 'y_up_to_z_up_flip_y':
            # Y-up to Z-up conversion + flip Y axis (flip new Y which was old Z)
            return x, -z, -y
        elif self.coordinate_transform == 'y_up_to_z_up_flip_z':
            # Y-up to Z-up conversion + flip Z axis (flip new Z which was old Y)
            return x, z, y
        elif self.coordinate_transform == 'y_up_to_z_up_flip_all':
            # Y-up to Z-up conversion + flip all axes
            return -x, -z, y
        elif self.coordinate_transform == 'y_up_to_z_up_flip_yx':
            # Y-up to Z-up conversion + flip Y and X axes
            return -x, -z, -y
        elif self.coordinate_transform == 'y_up_to_z_up_flip_yz':
            # Y-up to Z-up conversion + flip Y and Z axes
            return x, -z, y
        else:
            return x, y, z
    
    def _apply_scale_factor(self, x: float, y: float, z: float) -> tuple:
        """Apply uniform scaling to coordinates"""
        return x * self.scale_factor, y * self.scale_factor, z * self.scale_factor
    
    
    def detect_outliers(self, method: str = 'density', **kwargs) -> np.ndarray:
        """
        Detect outliers in the point cloud
        
        Args:
            method: 'density', 'statistical', or 'dbscan'
            **kwargs: Method-specific parameters
        
        Returns:
            Boolean array where True = inlier, False = outlier
        """
        if not self.points:
            raise ValueError("No points loaded")
        
        positions = np.array([[p.x, p.y, p.z] for p in self.points])
        
        # Additional NaN filtering for safety
        valid_mask = ~(np.isnan(positions).any(axis=1) | np.isinf(positions).any(axis=1))
        if not valid_mask.any():
            raise ValueError("No valid (non-NaN, non-infinite) points found")
        
        positions = positions[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        print(f"Filtered to {len(positions)} valid points (removed {len(self.points) - len(positions)} invalid points)")
        
        if method == 'none':
            print("No outlier detection - using all points")
            outlier_mask = np.ones(len(positions), dtype=bool)  # All points are inliers
        elif method == 'statistical':
            outlier_mask = self._statistical_outlier_removal(positions, **kwargs)
        elif method == 'dbscan':
            outlier_mask = self._dbscan_outlier_detection(positions, **kwargs)
        elif method == 'density':
            outlier_mask = self._density_based_outlier_detection(positions, **kwargs)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Map back to original point indices (accounting for filtered NaN points)
        full_outlier_mask = np.zeros(len(self.points), dtype=bool)
        full_outlier_mask[valid_indices] = outlier_mask
        
        return full_outlier_mask
    
    def _statistical_outlier_removal(self, positions: np.ndarray, 
                                   k_neighbors: int = 20, 
                                   std_ratio: float = 2.0) -> np.ndarray:
        """Statistical Outlier Removal (SOR)"""
        print(f"Running Statistical Outlier Removal (k={k_neighbors}, std_ratio={std_ratio})")
        
        # Find k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Calculate mean distance to neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Calculate global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # Mark points as inliers if within std_ratio standard deviations
        threshold = global_mean + std_ratio * global_std
        inliers = mean_distances < threshold
        
        outlier_count = np.sum(~inliers)
        print(f"SOR: Detected {outlier_count} outliers ({outlier_count/len(positions)*100:.1f}%)")
        
        return inliers
    
    def _dbscan_outlier_detection(self, positions: np.ndarray, 
                                eps: float = 0.5, 
                                min_samples: int = 10) -> np.ndarray:
        """DBSCAN-based outlier detection"""
        print(f"Running DBSCAN outlier detection (eps={eps}, min_samples={min_samples})")
        
        # Run DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(positions)
        
        # Points labeled as -1 are outliers
        inliers = labels != -1
        
        outlier_count = np.sum(~inliers)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"DBSCAN: Found {cluster_count} clusters, {outlier_count} outliers ({outlier_count/len(positions)*100:.1f}%)")
        
        return inliers
    
    def _density_based_outlier_detection(self, positions: np.ndarray, 
                                       k_neighbors: int = 20, 
                                       density_threshold: float = 0.1) -> np.ndarray:
        """Custom density-based outlier detection"""
        print(f"Running density-based outlier detection (k={k_neighbors}, threshold={density_threshold})")
        
        # Calculate local density for each point
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Calculate local density as inverse of mean distance to neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)
        local_densities = 1.0 / (mean_distances + 1e-10)  # Avoid division by zero
        
        # Normalize densities
        normalized_densities = local_densities / np.max(local_densities)
        
        # Keep points above density threshold
        inliers = normalized_densities >= density_threshold
        
        outlier_count = np.sum(~inliers)
        print(f"Density: Detected {outlier_count} outliers ({outlier_count/len(positions)*100:.1f}%)")
        
        return inliers
    
    def calculate_bounds(self, inlier_mask: np.ndarray, 
                        method: str = 'alpha_shape', 
                        **kwargs) -> List[Tuple[float, float, float]]:
        """
        Calculate 3D bounds of inlier points
        
        Args:
            inlier_mask: Boolean array indicating inlier points
            method: 'convex_hull', 'alpha_shape', or 'tight_bounds'
            **kwargs: Method-specific parameters
        
        Returns:
            List of (x, y, z) tuples defining the 3D bounding box vertices
        """
        if not self.points:
            raise ValueError("No points loaded")
        
        # Filter to inlier points only
        inlier_points = [p for i, p in enumerate(self.points) if inlier_mask[i]]
        positions = np.array([[p.x, p.y, p.z] for p in inlier_points])
        
        print(f"Calculating bounds for {len(inlier_points)} inlier points using {method}")
        
        if method == 'convex_hull':
            return self._convex_hull_3d_bounds(positions, **kwargs)
        elif method == 'tight_bounds':
            return self._tight_3d_bounds(positions, **kwargs)
        elif method == 'alpha_shape':
            return self._alpha_shape_3d_bounds(positions, **kwargs)
        elif method == 'polygonal_prism':
            return self._polygonal_prism_bounds(positions, **kwargs)
        else:
            raise ValueError(f"Unknown bounds calculation method: {method}")
    
    def _convex_hull_3d_bounds(self, positions: np.ndarray, **kwargs) -> List[Tuple[float, float, float]]:
        """Calculate 3D convex hull and project to bounding box"""
        print("Calculating 3D convex hull bounds")
        
        try:
            # Calculate 3D convex hull
            hull_3d = ConvexHull(positions)
            hull_vertices = positions[hull_3d.vertices]
            
            # Find the bounding box of the convex hull vertices
            min_x, max_x = np.min(hull_vertices[:, 0]), np.max(hull_vertices[:, 0])
            min_y, max_y = np.min(hull_vertices[:, 1]), np.max(hull_vertices[:, 1])
            min_z, max_z = np.min(hull_vertices[:, 2]), np.max(hull_vertices[:, 2])
            
            # Create 3D bounding box vertices (8 corners)
            bounds = [
                (min_x, min_y, min_z),  # Bottom face
                (max_x, min_y, min_z),
                (max_x, max_y, min_z),
                (min_x, max_y, min_z),
                (min_x, min_y, max_z),  # Top face
                (max_x, min_y, max_z),
                (max_x, max_y, max_z),
                (min_x, max_y, max_z)
            ]
            
            print(f"3D convex hull bounding box: {len(bounds)} vertices")
            return bounds
            
        except Exception as e:
            print(f"3D convex hull failed, falling back to tight bounds: {e}")
            # Fallback to tight bounds
            return self._tight_3d_bounds(positions, **kwargs)
    
    def _tight_3d_bounds(self, positions: np.ndarray, 
                        buffer_ratio: float = 0.02) -> List[Tuple[float, float, float]]:
        """Calculate tight 3D bounding box with optional buffer"""
        print(f"Calculating tight 3D bounding box (buffer={buffer_ratio*100}%)")
        
        # Find min/max in X, Y, and Z
        min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
        min_y, max_y = np.min(positions[:, 1]), np.max(positions[:, 1])
        min_z, max_z = np.min(positions[:, 2]), np.max(positions[:, 2])
        
        # Add buffer to all dimensions
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        x_buffer = x_range * buffer_ratio
        y_buffer = y_range * buffer_ratio
        z_buffer = z_range * buffer_ratio
        
        min_x -= x_buffer
        max_x += x_buffer
        min_y -= y_buffer
        max_y += y_buffer
        min_z -= z_buffer
        max_z += z_buffer
        
        # Create 3D bounding box vertices (8 corners of a cube)
        bounds = [
            (min_x, min_y, min_z),  # Bottom face
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),  # Top face
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z)
        ]
        
        print(f"3D bounding box: {x_range:.2f} x {y_range:.2f} x {z_range:.2f} units")
        return bounds
    
    def _alpha_shape_3d_bounds(self, positions: np.ndarray, alpha: float = 0.1) -> List[Tuple[float, float, float]]:
        """Calculate alpha shape bounds (simplified 3D implementation)"""
        print(f"Calculating 3D alpha shape bounds (alpha={alpha})")
        
        # Simplified alpha shape - using 3D convex hull as fallback
        # A full alpha shape implementation would require more complex algorithms
        print("Note: Using 3D convex hull as alpha shape approximation")
        return self._convex_hull_3d_bounds(positions)
    
    def _polygonal_prism_bounds(self, positions: np.ndarray, 
                               buffer_ratio: float = 0.02,
                               max_vertices: int = 8,
                               polygon_method: str = 'convex_hull',
                               alpha: float = 0.1) -> List[Tuple[float, float, float]]:
        """Calculate polygonal prism bounds with identical top and bottom polygons"""
        print(f"Calculating polygonal prism bounds (method={polygon_method}, max_vertices={max_vertices}, buffer={buffer_ratio*100}%)")
        
        # Validate max_vertices
        if max_vertices < 3:
            print("Warning: max_vertices must be at least 3, using 3")
            max_vertices = 3
        
        # Calculate Z-range for prism height
        min_z, max_z = self._calculate_height_range(positions, buffer_ratio)
        
        # Calculate XY polygon for base
        xy_polygon = self._calculate_xy_polygon(positions, polygon_method, max_vertices, alpha)
        
        # Create prism vertices: bottom polygon + top polygon
        bounds = []
        
        # Bottom polygon vertices (at min_z)
        for x, y in xy_polygon:
            bounds.append((x, y, min_z))
        
        # Top polygon vertices (at max_z)
        for x, y in xy_polygon:
            bounds.append((x, y, max_z))
        
        print(f"Polygonal prism: {len(xy_polygon)} vertices per base, {len(bounds)} total vertices")
        print(f"Height range: {min_z:.2f} to {max_z:.2f} (range: {max_z - min_z:.2f})")
        
        return bounds
    
    def _calculate_height_range(self, positions: np.ndarray, buffer_ratio: float) -> Tuple[float, float]:
        """Calculate Z-range for prism extrusion with buffer"""
        min_z = np.min(positions[:, 2])
        max_z = np.max(positions[:, 2])
        z_range = max_z - min_z
        z_buffer = z_range * buffer_ratio
        return min_z - z_buffer, max_z + z_buffer
    
    def _calculate_xy_polygon(self, positions: np.ndarray, method: str, max_vertices: int, alpha: float = 0.1) -> List[Tuple[float, float]]:
        """Calculate 2D polygon from XY projection of 3D points"""
        # Project all points to XY plane
        xy_points = positions[:, :2]  # Drop Z coordinate
        
        if method == 'convex_hull':
            return self._xy_convex_hull(xy_points, max_vertices)
        elif method == 'alpha_shape':
            return self._xy_alpha_shape(xy_points, max_vertices, alpha)
        elif method == 'simplified_polygon':
            return self._xy_simplified_polygon(xy_points, max_vertices, alpha)
        else:
            raise ValueError(f"Unknown polygon method: {method}")
    
    def _xy_convex_hull(self, xy_points: np.ndarray, max_vertices: int) -> List[Tuple[float, float]]:
        """Calculate 2D convex hull with optional vertex reduction"""
        print(f"Calculating 2D convex hull from {len(xy_points)} points")
        
        try:
            # Calculate 2D convex hull
            hull = ConvexHull(xy_points)
            hull_vertices = xy_points[hull.vertices]
            
            print(f"Initial convex hull has {len(hull_vertices)} vertices")
            
            # Reduce vertices if needed using Douglas-Peucker algorithm
            if len(hull_vertices) > max_vertices:
                hull_vertices = self._reduce_polygon_vertices(hull_vertices, max_vertices)
                print(f"Reduced to {len(hull_vertices)} vertices")
            
            # Convert to list of tuples
            polygon = [(float(x), float(y)) for x, y in hull_vertices]
            
            return polygon
            
        except Exception as e:
            print(f"2D convex hull failed: {e}")
            # Fallback to tight bounds rectangle
            min_x, max_x = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
            min_y, max_y = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    def _reduce_polygon_vertices(self, vertices: np.ndarray, max_vertices: int) -> np.ndarray:
        """Reduce polygon vertices using Douglas-Peucker algorithm"""
        print(f"Reducing polygon from {len(vertices)} to {max_vertices} vertices")
        
        if len(vertices) <= max_vertices:
            return vertices
        
        # Apply Douglas-Peucker algorithm with adaptive tolerance
        reduced_vertices = self._douglas_peucker_polygon(vertices, max_vertices)
        
        return reduced_vertices
    
    def _douglas_peucker_polygon(self, vertices: np.ndarray, max_vertices: int) -> np.ndarray:
        """Apply Douglas-Peucker algorithm to reduce polygon vertices"""
        # For closed polygons, we need to handle the connection between first and last vertex
        # Ensure polygon is closed
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])
        
        # Find optimal tolerance using binary search
        tolerance = self._find_optimal_tolerance(vertices, max_vertices)
        
        # Apply Douglas-Peucker with the optimal tolerance
        simplified = self._douglas_peucker_line(vertices, tolerance)
        
        # Remove the duplicate closing vertex if it was added
        if len(simplified) > 1 and np.array_equal(simplified[0], simplified[-1]):
            simplified = simplified[:-1]
        
        print(f"Douglas-Peucker reduced to {len(simplified)} vertices (tolerance: {tolerance:.6f})")
        return simplified
    
    def _find_optimal_tolerance(self, vertices: np.ndarray, max_vertices: int) -> float:
        """Find optimal tolerance for Douglas-Peucker to achieve target vertex count"""
        # Calculate range of distances to set search bounds
        distances = []
        for i in range(1, len(vertices) - 1):
            dist = self._point_to_line_distance(
                vertices[i], vertices[0], vertices[-1]
            )
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Binary search for optimal tolerance
        min_tol = 0.0
        max_tol = max(distances) * 2
        
        for _ in range(20):  # Max 20 iterations
            tolerance = (min_tol + max_tol) / 2
            simplified = self._douglas_peucker_line(vertices, tolerance)
            
            # Remove duplicate closing vertex for counting
            vertex_count = len(simplified)
            if vertex_count > 1 and np.array_equal(simplified[0], simplified[-1]):
                vertex_count -= 1
            
            if vertex_count <= max_vertices:
                max_tol = tolerance
            else:
                min_tol = tolerance
            
            # Close enough
            if abs(vertex_count - max_vertices) <= 1:
                break
        
        return max_tol
    
    def _douglas_peucker_line(self, vertices: np.ndarray, tolerance: float) -> np.ndarray:
        """Apply Douglas-Peucker algorithm to simplify a line"""
        if len(vertices) <= 2:
            return vertices
        
        # Find the point with maximum distance from line between first and last points
        max_dist = 0.0
        max_index = 0
        
        start_point = vertices[0]
        end_point = vertices[-1]
        
        for i in range(1, len(vertices) - 1):
            dist = self._point_to_line_distance(vertices[i], start_point, end_point)
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            # Recursively simplify the two segments
            left_simplified = self._douglas_peucker_line(vertices[:max_index + 1], tolerance)
            right_simplified = self._douglas_peucker_line(vertices[max_index:], tolerance)
            
            # Combine results (remove duplicate middle point)
            result = np.vstack([left_simplified[:-1], right_simplified])
            return result
        else:
            # All points are within tolerance, return only endpoints
            return np.array([start_point, end_point])
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line segment"""
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Handle degenerate case (line has zero length)
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq < 1e-10:
            return np.linalg.norm(point_vec)
        
        # Project point onto line
        projection = np.dot(point_vec, line_vec) / line_length_sq
        
        # Clamp projection to line segment
        projection = max(0.0, min(1.0, projection))
        
        # Find closest point on line segment
        closest_point = line_start + projection * line_vec
        
        # Return distance to closest point
        return np.linalg.norm(point - closest_point)
    
    def _xy_alpha_shape(self, xy_points: np.ndarray, max_vertices: int, alpha: float = 0.1) -> List[Tuple[float, float]]:
        """Calculate 2D alpha shape with optional vertex reduction"""
        print(f"Calculating 2D alpha shape from {len(xy_points)} points (alpha={alpha})")
        
        try:
            # Validate minimum points for triangulation
            if len(xy_points) < 3:
                print("Warning: Not enough points for alpha shape, using convex hull")
                return self._xy_convex_hull(xy_points, max_vertices)
            
            # Create Delaunay triangulation
            tri = Delaunay(xy_points)
            
            # Calculate alpha shape boundary
            alpha_vertices = self._compute_alpha_shape_boundary(xy_points, tri, alpha)
            
            if len(alpha_vertices) < 3:
                print("Warning: Alpha shape resulted in too few vertices, using convex hull")
                return self._xy_convex_hull(xy_points, max_vertices)
            
            print(f"Alpha shape has {len(alpha_vertices)} vertices")
            
            # Reduce vertices if needed
            if len(alpha_vertices) > max_vertices:
                alpha_vertices = self._reduce_polygon_vertices(alpha_vertices, max_vertices)
                print(f"Reduced to {len(alpha_vertices)} vertices")
            
            # Convert to list of tuples
            polygon = [(float(x), float(y)) for x, y in alpha_vertices]
            
            return polygon
            
        except Exception as e:
            print(f"Alpha shape calculation failed: {e}")
            print("Falling back to convex hull")
            return self._xy_convex_hull(xy_points, max_vertices)
    
    def _compute_alpha_shape_boundary(self, points: np.ndarray, tri: Delaunay, alpha: float) -> np.ndarray:
        """Compute alpha shape boundary from Delaunay triangulation"""
        print(f"Computing alpha shape boundary with alpha={alpha}")
        
        # Get all edges from triangulation
        edges = set()
        edge_triangles = {}  # Map edge to list of triangles that contain it
        
        for simplex in tri.simplices:
            # Each triangle has 3 edges
            triangle_edges = [
                tuple(sorted([simplex[0], simplex[1]])),
                tuple(sorted([simplex[1], simplex[2]])),
                tuple(sorted([simplex[2], simplex[0]]))
            ]
            
            for edge in triangle_edges:
                edges.add(edge)
                if edge not in edge_triangles:
                    edge_triangles[edge] = []
                edge_triangles[edge].append(simplex)
        
        # Filter edges based on alpha criterion
        alpha_edges = []
        
        for edge in edges:
            i, j = edge
            edge_length = np.linalg.norm(points[i] - points[j])
            
            # Get triangles that contain this edge
            triangles = edge_triangles[edge]
            
            if len(triangles) == 1:
                # Boundary edge - check if it's short enough
                if edge_length <= 2 * alpha:
                    alpha_edges.append(edge)
            else:
                # Interior edge - check circumradius criterion
                keep_edge = False
                for triangle in triangles:
                    circumradius = self._triangle_circumradius(points[triangle])
                    if circumradius <= alpha:
                        keep_edge = True
                        break
                
                if keep_edge:
                    alpha_edges.append(edge)
        
        print(f"Filtered to {len(alpha_edges)} alpha edges from {len(edges)} total edges")
        
        # Extract boundary polygon from alpha edges
        boundary_vertices = self._extract_boundary_from_edges(alpha_edges, points)
        
        return boundary_vertices
    
    def _triangle_circumradius(self, triangle_points: np.ndarray) -> float:
        """Calculate circumradius of a triangle"""
        # Triangle vertices
        a, b, c = triangle_points
        
        # Calculate side lengths
        side_a = np.linalg.norm(b - c)
        side_b = np.linalg.norm(c - a)
        side_c = np.linalg.norm(a - b)
        
        # Calculate area using cross product
        area = 0.5 * abs(np.cross(b - a, c - a))
        
        # Avoid division by zero
        if area < 1e-10:
            return float('inf')
        
        # Circumradius formula: R = (abc) / (4 * Area)
        circumradius = (side_a * side_b * side_c) / (4 * area)
        
        return circumradius
    
    def _extract_boundary_from_edges(self, edges: List[Tuple[int, int]], points: np.ndarray) -> np.ndarray:
        """Extract ordered boundary vertices from alpha shape edges"""
        if not edges:
            return np.array([])
        
        # Build adjacency graph
        graph = {}
        for edge in edges:
            i, j = edge
            if i not in graph:
                graph[i] = []
            if j not in graph:
                graph[j] = []
            graph[i].append(j)
            graph[j].append(i)
        
        # Find boundary cycle - start from any vertex with degree <= 2
        start_vertex = None
        for vertex, neighbors in graph.items():
            if len(neighbors) <= 2:
                start_vertex = vertex
                break
        
        if start_vertex is None:
            # All vertices have degree > 2, pick any
            start_vertex = list(graph.keys())[0]
        
        # Traverse boundary to get ordered vertices
        boundary = []
        current = start_vertex
        previous = None
        
        while True:
            boundary.append(current)
            
            # Find next vertex
            next_vertex = None
            for neighbor in graph[current]:
                if neighbor != previous:
                    next_vertex = neighbor
                    break
            
            if next_vertex is None or next_vertex == start_vertex:
                break
            
            previous = current
            current = next_vertex
            
            # Prevent infinite loops
            if len(boundary) > len(graph) * 2:
                break
        
        # Convert to numpy array of coordinates
        if boundary:
            boundary_coords = points[boundary]
            return boundary_coords
        else:
            return np.array([])
    
    def _xy_simplified_polygon(self, xy_points: np.ndarray, max_vertices: int, alpha: float = 0.1) -> List[Tuple[float, float]]:
        """Calculate simplified polygon using hybrid approach"""
        print(f"Calculating simplified polygon from {len(xy_points)} points (alpha={alpha})")
        
        try:
            # Step 1: Try alpha shape first for better boundary detection
            if len(xy_points) >= 3:
                try:
                    alpha_result = self._xy_alpha_shape(xy_points, max_vertices * 2, alpha)  # Allow more vertices initially
                    if len(alpha_result) >= 3:
                        print(f"Alpha shape successful with {len(alpha_result)} vertices")
                        
                        # Step 2: Apply additional simplification if needed
                        if len(alpha_result) > max_vertices:
                            # Convert back to numpy array for further processing
                            alpha_vertices = np.array(alpha_result)
                            
                            # Apply Douglas-Peucker to reduce vertices
                            simplified_vertices = self._douglas_peucker_polygon(alpha_vertices, max_vertices)
                            
                            # Convert back to list of tuples
                            result = [(float(x), float(y)) for x, y in simplified_vertices]
                            print(f"Simplified polygon: {len(result)} vertices after Douglas-Peucker")
                            return result
                        else:
                            print(f"Alpha shape already within target: {len(alpha_result)} vertices")
                            return alpha_result
                except Exception as e:
                    print(f"Alpha shape failed: {e}, trying convex hull approach")
            
            # Step 2: Fallback to convex hull with aggressive simplification
            print("Using convex hull fallback for simplified polygon")
            
            # Use convex hull as the base
            hull_result = self._xy_convex_hull(xy_points, max_vertices)
            
            if len(hull_result) > max_vertices:
                # Apply additional simplification
                hull_vertices = np.array(hull_result)
                simplified_vertices = self._douglas_peucker_polygon(hull_vertices, max_vertices)
                result = [(float(x), float(y)) for x, y in simplified_vertices]
                print(f"Simplified convex hull: {len(result)} vertices")
                return result
            else:
                print(f"Convex hull already within target: {len(hull_result)} vertices")
                return hull_result
            
        except Exception as e:
            print(f"Simplified polygon calculation failed: {e}")
            print("Using rectangular bounds as final fallback")
            # Final fallback to rectangular bounds
            min_x, max_x = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
            min_y, max_y = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

def create_cesium_clipping_polygon(bounds: List[Tuple[float, float, float]], 
                                 output_file: str = None,
                                 bounds_method: str = 'tight_bounds',
                                 polygon_method: str = 'convex_hull',
                                 alpha_value: float = 0.1) -> Dict:
    """
    Create Cesium-compatible clipping data from 3D bounds
    
    Args:
        bounds: List of (x, y, z) coordinate tuples representing 3D vertices
        output_file: Optional file to save JSON output
        bounds_method: Method used to calculate bounds
        polygon_method: Method used for polygon calculation (if applicable)
    
    Returns:
        Dictionary with 3D bounding box or polygonal prism data for Cesium transformation
    """
    
    # Store all 3D vertices - Cesium will transform and project these
    vertices_3d = []
    for x, y, z in bounds:
        vertices_3d.extend([x, y, z])
    
    # Determine output format based on bounds method
    if bounds_method == 'polygonal_prism':
        # For polygonal prism, we have N bottom vertices + N top vertices
        vertex_count = len(bounds)
        base_vertices = vertex_count // 2
        
        metadata = {
            "description": f"Generated polygonal prism from Gaussian splat bounds using {polygon_method}",
            "base_vertices": base_vertices,
            "total_vertices": vertex_count,
            "generated": True,
            "note": "3D vertices: first N are bottom polygon, next N are top polygon"
        }
        
        # Add alpha value to metadata if using alpha shape
        if polygon_method == 'alpha_shape':
            metadata["alpha_value"] = alpha_value
        
        clipping_data = {
            "vertices_3d": vertices_3d,
            "type": "polygonal_prism",
            "polygon_method": polygon_method,
            "vertex_count": vertex_count,
            "metadata": metadata
        }
    else:
        # Legacy 3D bounding box format
        clipping_data = {
            "vertices_3d": vertices_3d,
            "type": "3d_bounding_box",
            "metadata": {
                "description": "Generated 3D bounding box from Gaussian splat bounds",
                "vertexCount": len(bounds),
                "generated": True,
                "note": "3D vertices to be transformed by tileset matrix and projected to 2D in Cesium"
            }
        }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(clipping_data, f, indent=2)
        print(f"3D bounding box data saved to {output_file}")
    
    return clipping_data

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Process Gaussian splat files for bounds calculation')
    parser.add_argument('input_file', help='Input PLY or SPZ file')
    parser.add_argument('-o', '--output', help='Output JSON file for Cesium clipping polygon')
    parser.add_argument('--outlier-method', choices=['none', 'statistical', 'dbscan', 'density'], 
                       default='density', help='Outlier detection method')
    parser.add_argument('--bounds-method', choices=['convex_hull', 'tight_bounds', 'alpha_shape', 'polygonal_prism'], 
                       default='tight_bounds', help='Bounds calculation method')
    parser.add_argument('--k-neighbors', type=int, default=20, help='Number of neighbors for density calculations')
    parser.add_argument('--density-threshold', type=float, default=0.5, help='Density threshold for outlier detection')
    parser.add_argument('--buffer-ratio', type=float, default=0.02, help='Buffer ratio for tight bounds')
    parser.add_argument('--coordinate-transform', choices=['none', 'y_up_to_z_up', 'swap_yz', 'rdf_to_luf', 'luf_to_rdf', 'rub_to_rdf', 'flip_y', 'flip_z', 'flip_yz', 'y_up_to_z_up_flip_x', 'y_up_to_z_up_flip_y', 'y_up_to_z_up_flip_z', 'y_up_to_z_up_flip_all', 'y_up_to_z_up_flip_yx', 'y_up_to_z_up_flip_yz'], 
                       default='y_up_to_z_up_flip_z', help='Apply coordinate transformation for compatibility')
    parser.add_argument('--scale-factor', type=float, default=1.0, help='Scale factor to apply to all coordinates (e.g., 0.8 to shrink by 20%)')
    parser.add_argument('--max-vertices', type=int, default=8, help='Maximum vertices for polygonal prism base (minimum 3)')
    parser.add_argument('--polygon-method', choices=['convex_hull', 'alpha_shape', 'simplified_polygon'], default='convex_hull', help='Method for calculating polygonal base')
    parser.add_argument('--alpha-value', type=float, default=0.1, help='Alpha value for alpha shape calculation (smaller = more concave)')
    
    args = parser.parse_args()
    
    try:
        # Load and process the splat file
        processor = GaussianSplatProcessor(args.input_file, args.coordinate_transform, args.scale_factor)
        processor.load_points()
        
        # Detect outliers
        outlier_params = {'k_neighbors': args.k_neighbors}
        if args.outlier_method == 'density':
            outlier_params['density_threshold'] = args.density_threshold
        inlier_mask = processor.detect_outliers(method=args.outlier_method, **outlier_params)
        
        # Calculate bounds
        bounds_params = {
            'buffer_ratio': args.buffer_ratio
        }
        if args.bounds_method == 'polygonal_prism':
            bounds_params['max_vertices'] = args.max_vertices
            bounds_params['polygon_method'] = args.polygon_method
            bounds_params['alpha'] = args.alpha_value
        
        bounds = processor.calculate_bounds(inlier_mask, method=args.bounds_method, **bounds_params)
        
        # Create Cesium output
        output_file = args.output or "clipping-polygon.json"
        cesium_data = create_cesium_clipping_polygon(bounds, output_file, args.bounds_method, args.polygon_method, args.alpha_value)
        
        print(f"\nSuccess! Generated clipping polygon with {len(bounds)} vertices")
        print(f"Cesium clipping data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())