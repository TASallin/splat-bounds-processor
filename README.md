# Gaussian Splat Bounds Processor

A Python tool for extracting precise 3D bounding volumes from Gaussian Splat files (PLY/SPZ) with advanced outlier detection and coordinate system transformations. Supports both traditional rectangular prisms and polygonal prisms that better conform to splat geometry. Designed for integration with 3D rendering engines like Cesium for terrain clipping applications.

## Features

- **Multi-format Support**: Handles both PLY (ASCII/binary) and SPZ (compressed) Gaussian splat formats
- **Advanced Outlier Detection**: Multiple algorithms to filter sparse points and identify dense core regions
- **Flexible Bounding Volume Generation**: 
  - Traditional rectangular 3D bounding boxes
  - **Polygonal prisms** that better conform to splat geometry
  - Convex hull-based polygon calculation for optimal fit
- **Coordinate System Transformations**: Built-in support for various coordinate system conversions (RDF↔LUF, Y-up↔Z-up, etc.)
- **Configurable Polygon Complexity**: Control vertex count and polygon simplification
- **Flexible Scaling**: Apply uniform scaling to coordinates for fine-tuning alignment
- **Cesium Integration**: Outputs 3D bounding volume data ready for terrain clipping in Cesium applications

## Current Recommended Settings

```bash
python splat_bounds_processor.py your_splat.spz --bounds-method polygonal_prism --polygon-method convex_hull --max-vertices 40 --coordinate-transform y_up_to_z_up_flip_y --density-threshold 0.05
```

- Currently alpha_shape and simplified_polygon do not work, that can be fixed if we get a splat that is not represented well by a convex shape

## Installation

### Prerequisites

- Python 3.7+
- Required Python packages (install via pip)

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r splat_requirements.txt
```

**Required packages (`splat_requirements.txt`):**
```
numpy>=1.19.0
scikit-learn>=0.24.0
scipy>=1.6.0
```

## Quick Start

### Basic Usage

Process a Gaussian splat file with default settings (rectangular bounding box):

```bash
python splat_bounds_processor.py your_splat_file.spz
```

This will:
1. Load the splat file with Y-up to Z-up coordinate transformation
2. Detect outliers using density-based filtering (threshold 0.5)
3. Calculate a tight 3D bounding box
4. Output `clipping-polygon.json` with 3D bounding box data

### Polygonal Prism (Recommended)

Generate a polygonal prism that better conforms to splat geometry:

```bash
python splat_bounds_processor.py your_splat_file.spz --bounds-method polygonal_prism
```

This creates a prism with:
- Convex hull polygon base that fits the splat's 2D footprint
- Identical top and bottom polygons
- Height spanning the full Z-range of the splat data

For more precise boundaries that can follow concave shapes, use alpha shapes:

```bash
python splat_bounds_processor.py your_splat_file.spz \
  --bounds-method polygonal_prism \
  --polygon-method alpha_shape \
  --alpha-value 0.1
```

For optimal balance between accuracy and simplicity, use the simplified polygon method:

```bash
python splat_bounds_processor.py your_splat_file.spz \
  --bounds-method polygonal_prism \
  --polygon-method simplified_polygon \
  --alpha-value 0.1
```

### Common Use Cases

#### PLY File Processing (Rectangular)
```bash
python splat_bounds_processor.py input.ply \
  --coordinate-transform rdf_to_luf \
  --density-threshold 0.3
```

#### PLY File Processing (Polygonal)
```bash
python splat_bounds_processor.py input.ply \
  --bounds-method polygonal_prism \
  --coordinate-transform rdf_to_luf \
  --max-vertices 6
```

#### PLY File Processing (Alpha Shape)
```bash
python splat_bounds_processor.py input.ply \
  --bounds-method polygonal_prism \
  --polygon-method alpha_shape \
  --coordinate-transform rdf_to_luf \
  --alpha-value 0.08 \
  --max-vertices 10
```

#### PLY File Processing (Simplified Polygon)
```bash
python splat_bounds_processor.py input.ply \
  --bounds-method polygonal_prism \
  --polygon-method simplified_polygon \
  --coordinate-transform rdf_to_luf \
  --alpha-value 0.1 \
  --max-vertices 8
```

#### SPZ File Processing (Default Rectangular)
```bash
python splat_bounds_processor.py input.spz
# Uses: y_up_to_z_up_flip_z transform, density threshold 0.5
```

#### SPZ File Processing (Polygonal Prism)
```bash
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --max-vertices 8
```

#### Alpha Shape (Concave Polygons)
```bash
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method alpha_shape \
  --alpha-value 0.1 \
  --max-vertices 12
```

#### Simplified Polygon (Hybrid Approach)
```bash
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method simplified_polygon \
  --alpha-value 0.1 \
  --max-vertices 8
```

#### Conservative Outlier Removal
```bash
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --density-threshold 0.2 \
  --buffer-ratio 0.05
```

#### Custom Scaling and Transformation
```bash
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --coordinate-transform y_up_to_z_up_flip_y \
  --scale-factor 0.8 \
  --max-vertices 6
```

#### Alpha Shape Fine-tuning
```bash
# More detailed boundaries (smaller alpha)
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method alpha_shape \
  --alpha-value 0.05 \
  --max-vertices 16

# Simpler boundaries (larger alpha)
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method alpha_shape \
  --alpha-value 0.2 \
  --max-vertices 6
```

#### Simplified Polygon Examples
```bash
# Balanced approach (recommended for most cases)
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method simplified_polygon \
  --alpha-value 0.1 \
  --max-vertices 8

# Conservative simplification
python splat_bounds_processor.py input.spz \
  --bounds-method polygonal_prism \
  --polygon-method simplified_polygon \
  --alpha-value 0.05 \
  --max-vertices 12
```

## Command Line Reference

### Required Arguments

- `input_file`: Path to input PLY or SPZ file

### Optional Arguments

#### Output
- `-o, --output`: Output JSON file name (default: `clipping-polygon.json`)

#### Outlier Detection
- `--outlier-method`: Method for outlier detection
  - `none`: No outlier filtering (use all points)
  - `density`: Density-based filtering (default)
  - `statistical`: Statistical Outlier Removal (SOR)
  - `dbscan`: DBSCAN clustering-based detection
- `--density-threshold`: Density threshold for outlier detection (default: 0.5)
- `--k-neighbors`: Number of neighbors for density calculations (default: 20)

#### Bounds Calculation
- `--bounds-method`: Method for calculating bounds
  - `tight_bounds`: Rectangular 3D bounding box (default)
  - `convex_hull`: 3D convex hull bounding box
  - `alpha_shape`: Concave hull approximation
  - `polygonal_prism`: Polygonal prism with convex hull base (**recommended**)
- `--buffer-ratio`: Buffer percentage around bounds (default: 0.02 = 2%)

#### Polygonal Prism Options
- `--max-vertices`: Maximum vertices for polygonal base (default: 8, minimum: 3)
- `--polygon-method`: Method for calculating polygonal base
  - `convex_hull`: 2D convex hull of point projection (default)
  - `alpha_shape`: Concave alpha shape for better boundary fitting
  - `simplified_polygon`: Hybrid approach combining alpha shape with Douglas-Peucker simplification
- `--alpha-value`: Alpha parameter for alpha shape calculation (default: 0.1)
  - Smaller values create more concave (detailed) boundaries
  - Larger values create more convex (simplified) boundaries
  - Used by both `alpha_shape` and `simplified_polygon` methods

#### Coordinate Transformations
- `--coordinate-transform`: Coordinate system transformation (default: `y_up_to_z_up_flip_z`)
  - `none`: No transformation
  - `y_up_to_z_up`: Convert Y-up to Z-up coordinate system
  - `y_up_to_z_up_flip_z`: Y-up to Z-up + flip Z axis (recommended for SPZ)
  - `rdf_to_luf`: Convert RDF to LUF coordinate system
  - `luf_to_rdf`: Convert LUF back to RDF coordinate system
  - Additional variations: `flip_y`, `flip_z`, `flip_yz`, `y_up_to_z_up_flip_x`, etc.

#### Scaling
- `--scale-factor`: Uniform scale factor applied to all coordinates (default: 1.0)
  - Example: `0.8` shrinks coordinates by 20%
  - Example: `1.2` enlarges coordinates by 20%

## File Format Support

### Input Formats

#### PLY Format
- **ASCII PLY files**: Standard format from Gaussian splatting pipelines
- **Binary PLY files**: Supported with automatic encoding detection
- **Required properties**: x, y, z coordinates
- **Optional properties**: opacity, scale_0, scale_1, scale_2, colors, etc.
- **Coordinate systems**: Typically uses RDF (Right-Down-Front) convention

#### SPZ Format
- **Compressed format**: From Niantic Labs (~90% smaller than PLY)
- **Version support**: SPZ v2 with proper format specification
- **Accurate parsing**: Handles 24-bit fixed-point positions and header validation
- **Coordinate systems**: Uses RUB (Right-Up-Back, OpenGL/three.js) convention

### Output Format

The tool generates a JSON file with 3D bounding volume data for use in 3D applications:

#### Rectangular Bounding Box (Legacy Format)
```json
{
  "vertices_3d": [
    x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4,
    x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8
  ],
  "type": "3d_bounding_box",
  "metadata": {
    "description": "Generated 3D bounding box from Gaussian splat bounds",
    "vertexCount": 8,
    "generated": true,
    "note": "3D vertices to be transformed by application matrix and projected as needed"
  }
}
```

#### Polygonal Prism Format
```json
{
  "vertices_3d": [
    x1, y1, z_min, x2, y2, z_min, ..., xN, yN, z_min,
    x1, y1, z_max, x2, y2, z_max, ..., xN, yN, z_max
  ],
  "type": "polygonal_prism",
  "polygon_method": "convex_hull",
  "vertex_count": 12,
  "metadata": {
    "description": "Generated polygonal prism from Gaussian splat bounds using convex_hull",
    "base_vertices": 6,
    "total_vertices": 12,
    "generated": true,
    "note": "3D vertices: first N are bottom polygon, next N are top polygon"
  }
}
```

#### Alpha Shape Format
```json
{
  "vertices_3d": [
    x1, y1, z_min, x2, y2, z_min, ..., xN, yN, z_min,
    x1, y1, z_max, x2, y2, z_max, ..., xN, yN, z_max
  ],
  "type": "polygonal_prism",
  "polygon_method": "alpha_shape",
  "vertex_count": 16,
  "metadata": {
    "description": "Generated polygonal prism from Gaussian splat bounds using alpha_shape",
    "base_vertices": 8,
    "total_vertices": 16,
    "alpha_value": 0.1,
    "generated": true,
    "note": "3D vertices: first N are bottom polygon, next N are top polygon"
  }
}
```

#### Simplified Polygon Format
```json
{
  "vertices_3d": [
    x1, y1, z_min, x2, y2, z_min, ..., xN, yN, z_min,
    x1, y1, z_max, x2, y2, z_max, ..., xN, yN, z_max
  ],
  "type": "polygonal_prism",
  "polygon_method": "simplified_polygon",
  "vertex_count": 12,
  "metadata": {
    "description": "Generated polygonal prism from Gaussian splat bounds using simplified_polygon",
    "base_vertices": 6,
    "total_vertices": 12,
    "alpha_value": 0.1,
    "generated": true,
    "note": "3D vertices: first N are bottom polygon, next N are top polygon"
  }
}
```

**Vertex Order**:
- **Rectangular**: 8 vertices representing box corners (bottom face 0-3, top face 4-7)
- **Polygonal**: 2N vertices representing identical top and bottom polygons
  - First N vertices: Bottom polygon (counterclockwise when viewed from above)
  - Next N vertices: Top polygon (same order as bottom)

**Algorithm Summary**:
- **Convex Hull**: Simple, fast, always convex boundaries
- **Alpha Shape**: Detailed concave boundaries, best for complex shapes
- **Simplified Polygon**: Hybrid approach using alpha shape + Douglas-Peucker simplification for optimal balance

## Outlier Detection Methods

### Density-Based Detection (Recommended)
Removes points with low local density relative to their neighbors.

**Parameters:**
- `--density-threshold`: Points below this density ratio are removed (0.1-0.9)
- `--k-neighbors`: Number of neighbors for density calculation (10-50)

**Best for**: Gaussian splats with dense core regions surrounded by sparse outliers

### Statistical Outlier Removal (SOR)
Removes points based on statistical analysis of neighbor distances.

**Parameters:**
- `--k-neighbors`: Number of neighbors to analyze (10-50)
- Standard deviation ratio is fixed at 2.0

**Best for**: Uniformly distributed outliers with consistent noise patterns

### DBSCAN Clustering
Marks unclustered points as outliers using density-based spatial clustering.

**Parameters:**
- Automatically determines parameters based on point density

**Best for**: Complex outlier patterns with multiple dense clusters

## Coordinate System Transformations

### Common Transformations

#### For PLY Files
```bash
--coordinate-transform rdf_to_luf
```
Converts from RDF (Right-Down-Front) to LUF (Left-Up-Front) coordinate system.

#### For SPZ Files (Default)
```bash
--coordinate-transform y_up_to_z_up_flip_z
```
Converts Y-up to Z-up coordinate system and flips Z axis for proper orientation.

#### Custom Transformations
- `y_up_to_z_up`: Basic Y-up to Z-up conversion
- `flip_y`, `flip_z`, `flip_yz`: Simple axis flips
- `y_up_to_z_up_flip_x`: Y-up to Z-up + flip X axis
- `y_up_to_z_up_flip_y`: Y-up to Z-up + flip Y axis
- `y_up_to_z_up_flip_all`: Y-up to Z-up + flip all axes

## Integration Examples

### Cesium Terrain Clipping

Place the generated JSON file in your data directory:
```
/data/
  /your-site-id/
    tileset.json
    content.glb
    clipping-polygon.json  <- Generated file
```

The 3D bounding volume (rectangular or polygonal) can be transformed and projected to 2D for use with Cesium's ClippingPolygon system. Polygonal prisms provide more accurate clipping boundaries that better match the actual splat geometry.

### Web Applications

Load the JSON data in JavaScript:
```javascript
fetch('clipping-polygon.json')
  .then(response => response.json())
  .then(data => {
    const vertices3D = data.vertices_3d;
    
    if (data.type === 'polygonal_prism') {
      const baseVertices = data.metadata.base_vertices;
      console.log(`Processing ${baseVertices}-sided polygonal prism`);
      // Process N*2 vertices (N bottom + N top)
    } else {
      console.log('Processing rectangular bounding box');
      // Process 8 vertices (24 coordinates total)
    }
  });
```

## Performance Considerations

- **Large Files**: Files with >2M points may take several minutes to process
- **Memory Usage**: Dense point clouds require significant RAM
- **Processing Speed**: Outlier detection is the most time-intensive step
- **Polygon Complexity**: Higher `--max-vertices` values increase processing time slightly
- **Coordinate Systems**: Choose the right transformation to avoid post-processing
- **Polygonal vs Rectangular**: Polygonal prism calculation adds minimal overhead

## Troubleshooting

### Common Issues

1. **Bounding box doesn't align with splat**:
   - Try different coordinate transformations (`--coordinate-transform`)
   - Check if PLY and SPZ use different coordinate systems
   - Verify the coordinate system expected by your 3D application

2. **Bounding box too large**:
   - Increase density threshold (`--density-threshold 0.7`)
   - Try statistical outlier method (`--outlier-method statistical`)
   - Reduce buffer ratio (`--buffer-ratio 0.01`)

3. **Bounding box too small**:
   - Decrease density threshold (`--density-threshold 0.2`)
   - Increase buffer ratio (`--buffer-ratio 0.05`)
   - Use `--outlier-method none` to include all points

4. **File parsing errors**:
   - Ensure PLY/SPZ file is not corrupted
   - Try different coordinate transformations
   - For SPZ files, verify they were created with compatible tools

5. **Scale/position issues**:
   - Use `--scale-factor` to adjust coordinate scaling
   - Try different coordinate transformations
   - Check if your 3D application expects specific coordinate ranges

### Debug Output

The tool provides detailed output including:
- Raw coordinate values and transformations applied
- Coordinate ranges and distribution analysis
- Outlier detection statistics
- Final bounding box dimensions

## Advanced Usage

### Batch Processing
```bash
for file in *.spz; do
    python splat_bounds_processor.py "$file" \
      --bounds-method polygonal_prism \
      --output "${file%.spz}_clipping.json" \
      --density-threshold 0.6
done
```

### Fine-tuning for Specific Applications
```bash
# For architectural models (precise bounds)
python splat_bounds_processor.py model.spz \
  --bounds-method polygonal_prism \
  --outlier-method statistical \
  --k-neighbors 50 \
  --buffer-ratio 0.01 \
  --max-vertices 12

# For landscape models (include more area)
python splat_bounds_processor.py landscape.spz \
  --bounds-method polygonal_prism \
  --density-threshold 0.3 \
  --buffer-ratio 0.1 \
  --max-vertices 8 \
  --scale-factor 1.1

# For simple/fast processing
python splat_bounds_processor.py simple.spz \
  --bounds-method polygonal_prism \
  --max-vertices 4 \
  --outlier-method none
```

## Development

### Project Structure
```
splat_bounds_processor.py    # Main processing script
splat_requirements.txt       # Python dependencies
README.md                    # This documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### License

This project is open source. Please check the license file for specific terms.

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information about your use case

Include the following in bug reports:
- Input file format and size
- Command line arguments used
- Complete error output
- Expected vs actual behavior