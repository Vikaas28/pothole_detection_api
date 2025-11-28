from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import base64
from datetime import datetime
from typing import List, Dict, Any
import json
import shutil
from io import BytesIO
from PIL import Image
import os

app = FastAPI(title="Pothole Detection System", version="2.0.0")

# Configuration
MODEL_PATH = r"C:\Users\virat\OneDrive\Desktop\pothole\best.pt"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
CONFIDENCE_THRESHOLD = 0.25

# Create necessary directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Load YOLO model with a safe fallback
model = None
try:
    if not Path(MODEL_PATH).exists():
        print(f"Model file not found at {MODEL_PATH}. Trying fallback 'yolov8n.pt'...")
        fallback_model = "yolov8n.pt"
        model = YOLO(fallback_model)
        print(f"Fallback model loaded successfully: {fallback_model}")
    else:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


class MaterialEstimator:
    """Calculate material requirements for pothole repairs"""
    
    # Material specifications (industry standards)
    MATERIALS = {
        'cold_mix_asphalt': {
            'name': 'Cold Mix Asphalt',
            'density': 2400,  # kg/m³
            'unit_price': 85,  # $ per ton
            'wastage': 1.10  # 10% wastage factor
        },
        'hot_mix_asphalt': {
            'name': 'Hot Mix Asphalt',
            'density': 2350,  # kg/m³
            'unit_price': 95,  # $ per ton
            'wastage': 1.08
        },
        'tack_coat': {
            'name': 'Tack Coat Emulsion',
            'coverage': 0.3,  # liters per m²
            'unit_price': 2.5,  # $ per liter
            'wastage': 1.05
        },
        'concrete_mix': {
            'name': 'Concrete Patch Mix',
            'density': 2400,  # kg/m³
            'unit_price': 120,  # $ per ton
            'wastage': 1.12
        },
        'prime_coat': {
            'name': 'Prime Coat',
            'coverage': 0.5,  # liters per m²
            'unit_price': 3.0,  # $ per liter
            'wastage': 1.05
        },
        'aggregate_base': {
            'name': 'Aggregate Base Course',
            'density': 1800,  # kg/m³
            'unit_price': 45,  # $ per ton
            'wastage': 1.15
        },
        'seal_coat': {
            'name': 'Seal Coat',
            'coverage': 0.4,  # liters per m²
            'unit_price': 2.8,  # $ per liter
            'wastage': 1.05
        }
    }
    
    # Repair depth based on severity (in meters)
    DEPTH_SPECIFICATIONS = {
        'high': {
            'total_depth': 0.15,  # 15cm total depth
            'base_depth': 0.10,   # 10cm base layer
            'surface_depth': 0.05  # 5cm surface layer
        },
        'medium': {
            'total_depth': 0.10,
            'base_depth': 0.06,
            'surface_depth': 0.04
        },
        'low': {
            'total_depth': 0.05,
            'base_depth': 0.03,
            'surface_depth': 0.02
        }
    }
    
    @staticmethod
    def pixels_to_meters(pixel_area: float, image_width_px: int, assumed_road_width_m: float = 3.5) -> float:
        """Convert pixel area to square meters"""
        meters_per_pixel = assumed_road_width_m / image_width_px
        area_m2 = pixel_area * (meters_per_pixel ** 2)
        return area_m2
    
    @staticmethod
    def calculate_materials(detections: List[Dict], image_shape: tuple) -> Dict:
        """Calculate comprehensive material requirements"""
        
        if not detections:
            return MaterialEstimator._empty_materials()
        
        image_width_px = image_shape[1]
        severity_data = {'high': [], 'medium': [], 'low': []}
        
        # Group detections by severity and convert to m²
        for det in detections:
            area_m2 = MaterialEstimator.pixels_to_meters(det['area'], image_width_px)
            severity_data[det['severity']].append(area_m2)
        
        # Calculate totals
        total_area_m2 = sum(sum(areas) for areas in severity_data.values())
        severity_totals = {sev: sum(areas) for sev, areas in severity_data.items()}
        
        materials = {}
        total_cost = 0
        
        # 1. AGGREGATE BASE COURSE
        base_volume = 0
        for severity, area in severity_totals.items():
            if area > 0:
                depth = MaterialEstimator.DEPTH_SPECIFICATIONS[severity]['base_depth']
                base_volume += area * depth
        
        if base_volume > 0:
            agg = MaterialEstimator.MATERIALS['aggregate_base']
            base_weight = (base_volume * agg['density'] * agg['wastage']) / 1000  # tons
            base_cost = base_weight * agg['unit_price']
            materials['aggregate_base'] = {
                'name': agg['name'],
                'quantity': round(base_weight, 2),
                'unit': 'tons',
                'cost': round(base_cost, 2),
                'description': 'Foundation layer for structural support'
            }
            total_cost += base_cost
        
        # 2. PRIME COAT
        if total_area_m2 > 0:
            prime = MaterialEstimator.MATERIALS['prime_coat']
            prime_volume = total_area_m2 * prime['coverage'] * prime['wastage']
            prime_cost = prime_volume * prime['unit_price']
            materials['prime_coat'] = {
                'name': prime['name'],
                'quantity': round(prime_volume, 2),
                'unit': 'liters',
                'cost': round(prime_cost, 2),
                'description': 'Waterproofing and bonding layer'
            }
            total_cost += prime_cost
        
        # 3. TACK COAT
        if total_area_m2 > 0:
            tack = MaterialEstimator.MATERIALS['tack_coat']
            tack_volume = total_area_m2 * tack['coverage'] * tack['wastage']
            tack_cost = tack_volume * tack['unit_price']
            materials['tack_coat'] = {
                'name': tack['name'],
                'quantity': round(tack_volume, 2),
                'unit': 'liters',
                'cost': round(tack_cost, 2),
                'description': 'Adhesive layer between base and surface'
            }
            total_cost += tack_cost
        
        # 4. COLD MIX ASPHALT (Surface Layer)
        surface_volume = 0
        for severity, area in severity_totals.items():
            if area > 0:
                depth = MaterialEstimator.DEPTH_SPECIFICATIONS[severity]['surface_depth']
                surface_volume += area * depth
        
        if surface_volume > 0:
            asphalt = MaterialEstimator.MATERIALS['cold_mix_asphalt']
            asphalt_weight = (surface_volume * asphalt['density'] * asphalt['wastage']) / 1000
            asphalt_cost = asphalt_weight * asphalt['unit_price']
            materials['cold_mix_asphalt'] = {
                'name': asphalt['name'],
                'quantity': round(asphalt_weight, 2),
                'unit': 'tons',
                'cost': round(asphalt_cost, 2),
                'description': 'Primary patching material for all-weather repair'
            }
            total_cost += asphalt_cost
        
        # 5. HOT MIX ASPHALT (Alternative - for high severity only)
        high_surface_volume = 0
        if severity_totals['high'] > 0:
            depth = MaterialEstimator.DEPTH_SPECIFICATIONS['high']['surface_depth']
            high_surface_volume = severity_totals['high'] * depth
            
            hot_asphalt = MaterialEstimator.MATERIALS['hot_mix_asphalt']
            hot_weight = (high_surface_volume * hot_asphalt['density'] * hot_asphalt['wastage']) / 1000
            hot_cost = hot_weight * hot_asphalt['unit_price']
            materials['hot_mix_asphalt'] = {
                'name': hot_asphalt['name'],
                'quantity': round(hot_weight, 2),
                'unit': 'tons',
                'cost': round(hot_cost, 2),
                'description': 'Premium option for high-severity repairs'
            }
        
        # 6. SEAL COAT
        if total_area_m2 > 0:
            seal = MaterialEstimator.MATERIALS['seal_coat']
            seal_volume = total_area_m2 * seal['coverage'] * seal['wastage']
            seal_cost = seal_volume * seal['unit_price']
            materials['seal_coat'] = {
                'name': seal['name'],
                'quantity': round(seal_volume, 2),
                'unit': 'liters',
                'cost': round(seal_cost, 2),
                'description': 'Final protective and waterproof coating'
            }
            total_cost += seal_cost
        
        # Calculate labor and equipment costs (typically 40-50% of material cost)
        labor_cost = total_cost * 0.45
        
        return {
            'materials': materials,
            'area_summary': {
                'total_area_m2': round(total_area_m2, 2),
                'total_area_sqft': round(total_area_m2 * 10.764, 2),
                'high_severity_m2': round(severity_totals['high'], 2),
                'medium_severity_m2': round(severity_totals['medium'], 2),
                'low_severity_m2': round(severity_totals['low'], 2)
            },
            'cost_summary': {
                'materials_cost': round(total_cost, 2),
                'labor_cost': round(labor_cost, 2),
                'equipment_cost': round(total_cost * 0.15, 2),
                'total_project_cost': round(total_cost * 1.60, 2),
                'currency': 'USD'
            },
            'depth_specifications': MaterialEstimator.DEPTH_SPECIFICATIONS
        }
    
    @staticmethod
    def _empty_materials():
        """Return empty material estimation"""
        return {
            'materials': {},
            'area_summary': {
                'total_area_m2': 0,
                'total_area_sqft': 0,
                'high_severity_m2': 0,
                'medium_severity_m2': 0,
                'low_severity_m2': 0
            },
            'cost_summary': {
                'materials_cost': 0,
                'labor_cost': 0,
                'equipment_cost': 0,
                'total_project_cost': 0,
                'currency': 'USD'
            }
        }


class PotholeDetector:
    """Main class for pothole detection and analysis"""
    
    def __init__(self, model):
        self.model = model
        self.colors = {
            'high': (220, 53, 69),
            'medium': (255, 193, 7),
            'low': (40, 167, 69),
            'box': (0, 123, 255),
            'text_bg': (255, 255, 255),
            'text': (33, 37, 41)
        }
    
    def calculate_severity(self, area: float, img_area: float) -> str:
        """Calculate pothole severity based on relative area"""
        relative_area = (area / img_area) * 100
        
        if relative_area > 5:
            return 'high'
        elif relative_area > 2:
            return 'medium'
        else:
            return 'low'
    
    def draw_detections(self, image: np.ndarray, results: Any) -> tuple:
        """Draw bounding boxes and labels on image"""
        detections = []
        img_area = image.shape[0] * image.shape[1]
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                width = x2 - x1
                height = y2 - y1
                area = width * height
                severity = self.calculate_severity(area, img_area)
                
                box_color = self.colors[severity]
                
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
                
                label = f"Pothole {confidence:.2f}"
                severity_label = f"{severity.upper()}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                (sev_w, sev_h), _ = cv2.getTextSize(severity_label, font, font_scale - 0.1, thickness)
                
                cv2.rectangle(image, (x1, y1 - text_h - 35), 
                            (x1 + max(text_w, sev_w) + 10, y1), 
                            box_color, -1)
                
                cv2.putText(image, label, (x1 + 5, y1 - text_h - 15), 
                           font, font_scale, self.colors['text_bg'], thickness)
                cv2.putText(image, severity_label, (x1 + 5, y1 - 5), 
                           font, font_scale - 0.1, self.colors['text_bg'], thickness - 1)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'area': area,
                    'severity': severity,
                    'width': width,
                    'height': height
                })
        
        return image, detections
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process image and detect potholes"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_image = image.copy()
        
        results = self.model(image, conf=CONFIDENCE_THRESHOLD)
        
        annotated_image, detections = self.draw_detections(image, results)
        
        output_filename = f"detected_{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), annotated_image)
        
        stats = self.calculate_statistics(detections, original_image.shape)
        material_estimate = MaterialEstimator.calculate_materials(detections, original_image.shape)
        
        return {
            'output_path': str(output_path),
            'detections': detections,
            'statistics': stats,
            'material_estimate': material_estimate,
            'image_shape': original_image.shape
        }
    
    def calculate_statistics(self, detections: List[Dict], image_shape: tuple) -> Dict:
        """Calculate detection statistics"""
        total_detections = len(detections)
        
        if total_detections == 0:
            return {
                'total_potholes': 0,
                'severity_counts': {'high': 0, 'medium': 0, 'low': 0},
                'avg_confidence': 0,
                'total_area': 0,
                'coverage_percentage': 0
            }
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        total_confidence = 0
        total_area = 0
        
        for det in detections:
            severity_counts[det['severity']] += 1
            total_confidence += det['confidence']
            total_area += det['area']
        
        img_area = image_shape[0] * image_shape[1]
        
        return {
            'total_potholes': total_detections,
            'severity_counts': severity_counts,
            'avg_confidence': total_confidence / total_detections,
            'total_area': total_area,
            'coverage_percentage': (total_area / img_area) * 100
        }


class ReportGenerator:
    """Generate enhanced HTML reports"""
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    @staticmethod
    def generate_html_report(result: Dict, original_filename: str) -> str:
        """Generate comprehensive HTML report with material estimation"""
        stats = result['statistics']
        detections = result['detections']
        materials = result['material_estimate']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        output_image_b64 = ReportGenerator.encode_image(result['output_path'])
        
        # Generate materials table HTML
        materials_html = ""
        if materials['materials']:
            for mat_key, mat_data in materials['materials'].items():
                materials_html += f"""
                <tr>
                    <td><strong>{mat_data['name']}</strong></td>
                    <td>{mat_data['quantity']} {mat_data['unit']}</td>
                    <td>${mat_data['cost']:,.2f}</td>
                    <td><span class="material-desc">{mat_data['description']}</span></td>
                </tr>
                """
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Detection & Repair Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
            padding: 20px;
            min-height: 100vh;
            color: #1f2937;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 50px 60px;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
            border-radius: 50%;
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        
        .header .subtitle {{
            font-size: 1.25em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .metadata {{
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 35px 60px;
            border-bottom: 1px solid #cbd5e1;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
        }}
        
        .metadata-item {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metadata-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }}
        
        .metadata-icon {{
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            flex-shrink: 0;
        }}
        
        .metadata-content {{
            flex: 1;
        }}
        
        .metadata-label {{
            font-size: 0.8em;
            color: #64748b;
            margin-bottom: 4px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metadata-value {{
            font-size: 1.1em;
            font-weight: 700;
            color: #1e293b;
        }}
        
        .content {{
            padding: 50px 60px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 2em;
            color: #1e293b;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 4px solid #6366f1;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .section-icon {{
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            color: white;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin-bottom: 35px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 30px;
            border-radius: 16px;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            border-color: #6366f1;
        }}
        
        .stat-value {{
            font-size: 2.8em;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 10px;
            line-height: 1;
        }}
        
        .stat-label {{
            color: #64748b;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
        }}
        
        .severity-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 35px;
        }}
        
        .severity-card {{
            padding: 35px;
            border-radius: 16px;
            text-align: center;
            color: white;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .severity-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
        }}
        
        .severity-card:hover {{
            transform: scale(1.05);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        
        .severity-high {{
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        }}
        
        .severity-medium {{
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }}
        
        .severity-low {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }}
        
        .severity-count {{
            font-size: 3.5em;
            font-weight: 800;
            margin-bottom: 12px;
            position: relative;
            z-index: 1;
        }}
        
        .severity-label {{
            font-size: 1.15em;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
            position: relative;
            z-index: 1;
        }}
        
        .image-container {{
            background: #f8fafc;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 35px;
            border: 2px solid #e2e8f0;
        }}
        
        .image-container img {{
            width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
        }}
        
        .materials-section {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 35px;
            border: 3px solid #fbbf24;
        }}
        
        .cost-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .cost-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .cost-value {{
            font-size: 2.2em;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 8px;
        }}
        
        .cost-label {{
            font-size: 0.9em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        
        .total-cost {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
        }}
        
        .total-cost .cost-value {{
            color: white;
        }}
        
        .total-cost .cost-label {{
            color: rgba(255, 255, 255, 0.9);
        }}
        
        .materials-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .materials-table thead {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
        }}
        
        .materials-table th {{
            padding: 20px;
            text-align: left;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-size: 0.9em;
        }}
        
        .materials-table td {{
            padding: 18px 20px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .materials-table tbody tr:hover {{
            background: #f8fafc;
        }}
        
        .material-desc {{
            color: #64748b;
            font-size: 0.9em;
            font-style: italic;
        }}
        
        .detections-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .detections-table thead {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
        }}
        
        .detections-table th {{
            padding: 20px;
            text-align: left;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-size: 0.9em;
        }}
        
        .detections-table td {{
            padding: 18px 20px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .detections-table tbody tr:hover {{
            background: #f8fafc;
        }}
        
        .severity-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .badge-high {{
            background: #dc2626;
            color: white;
        }}
        
        .badge-medium {{
            background: #f59e0b;
            color: white;
        }}
        
        .badge-low {{
            background: #10b981;
            color: white;
        }}
        
        .confidence-bar {{
            width: 100%;
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            transition: width 0.3s ease;
        }}
        
        .area-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .area-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #6366f1;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .area-value {{
            font-size: 1.8em;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 5px;
        }}
        
        .area-label {{
            font-size: 0.85em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        
        .footer {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 40px 60px;
            text-align: center;
        }}
        
        .footer p {{
            margin: 8px 0;
            opacity: 0.9;
        }}
        
        .footer strong {{
            font-weight: 700;
            font-size: 1.1em;
        }}
        
        .info-box {{
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .info-box-title {{
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .info-box-content {{
            color: #1e3a8a;
            line-height: 1.6;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 30px 30px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .content {{
                padding: 30px 30px;
            }}
            
            .metadata {{
                padding: 25px 30px;
            }}
            
            .stats-grid, .severity-grid, .cost-summary {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
            
            .stat-card:hover, .severity-card:hover {{
                transform: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>🛣️ Pothole Detection & Repair Analysis Report</h1>
                <p class="subtitle">Comprehensive AI-Powered Road Condition Assessment with Material Estimation</p>
            </div>
        </div>
        
        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-icon">📁</div>
                    <div class="metadata-content">
                        <div class="metadata-label">File Name</div>
                        <div class="metadata-value">{original_filename}</div>
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-icon">📅</div>
                    <div class="metadata-content">
                        <div class="metadata-label">Analysis Date</div>
                        <div class="metadata-value">{timestamp}</div>
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-icon">🤖</div>
                    <div class="metadata-content">
                        <div class="metadata-label">Detection Model</div>
                        <div class="metadata-value">YOLOv8</div>
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-icon">🎯</div>
                    <div class="metadata-content">
                        <div class="metadata-label">Confidence Threshold</div>
                        <div class="metadata-value">{CONFIDENCE_THRESHOLD * 100}%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">
                    <div class="section-icon">📊</div>
                    Detection Summary
                </h2>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_potholes']}</div>
                        <div class="stat-label">Total Potholes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['avg_confidence']:.1%}</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['coverage_percentage']:.2f}%</div>
                        <div class="stat-label">Coverage Area</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_area']:,}</div>
                        <div class="stat-label">Total Area (px²)</div>
                    </div>
                </div>
                
                <div class="severity-grid">
                    <div class="severity-card severity-high">
                        <div class="severity-count">{stats['severity_counts']['high']}</div>
                        <div class="severity-label">🔴 High Severity</div>
                    </div>
                    <div class="severity-card severity-medium">
                        <div class="severity-count">{stats['severity_counts']['medium']}</div>
                        <div class="severity-label">🟡 Medium Severity</div>
                    </div>
                    <div class="severity-card severity-low">
                        <div class="severity-count">{stats['severity_counts']['low']}</div>
                        <div class="severity-label">🟢 Low Severity</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <div class="section-icon">🖼️</div>
                    Detection Visualization
                </h2>
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{output_image_b64}" alt="Detection Result">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <div class="section-icon">🔨</div>
                    Material Estimation & Repair Cost Analysis
                </h2>
                
                <div class="info-box">
                    <div class="info-box-title">📏 Estimation Methodology</div>
                    <div class="info-box-content">
                        Material calculations are based on detected pothole areas with assumed road width of 3.5 meters. 
                        Repair depths vary by severity: High (15cm), Medium (10cm), Low (5cm). 
                        All estimates include industry-standard wastage factors and follow ASTM standards for road repair.
                    </div>
                </div>
                
                <div class="area-breakdown">
                    <div class="area-card">
                        <div class="area-value">{materials['area_summary']['total_area_m2']} m²</div>
                        <div class="area-label">Total Repair Area</div>
                    </div>
                    <div class="area-card">
                        <div class="area-value">{materials['area_summary']['total_area_sqft']} ft²</div>
                        <div class="area-label">Total Area (Imperial)</div>
                    </div>
                    <div class="area-card">
                        <div class="area-value">{materials['area_summary']['high_severity_m2']} m²</div>
                        <div class="area-label">High Severity Area</div>
                    </div>
                    <div class="area-card">
                        <div class="area-value">{materials['area_summary']['medium_severity_m2']} m²</div>
                        <div class="area-label">Medium Severity Area</div>
                    </div>
                </div>
                
                <div class="materials-section">
                    <h3 style="margin-bottom: 25px; font-size: 1.5em; color: #78350f;">💰 Cost Summary</h3>
                    <div class="cost-summary">
                        <div class="cost-card">
                            <div class="cost-value">${materials['cost_summary']['materials_cost']:,.2f}</div>
                            <div class="cost-label">Materials Cost</div>
                        </div>
                        <div class="cost-card">
                            <div class="cost-value">${materials['cost_summary']['labor_cost']:,.2f}</div>
                            <div class="cost-label">Labor Cost</div>
                        </div>
                        <div class="cost-card">
                            <div class="cost-value">${materials['cost_summary']['equipment_cost']:,.2f}</div>
                            <div class="cost-label">Equipment Cost</div>
                        </div>
                        <div class="cost-card total-cost">
                            <div class="cost-value">${materials['cost_summary']['total_project_cost']:,.2f}</div>
                            <div class="cost-label">Total Project Cost</div>
                        </div>
                    </div>
                    
                    <h3 style="margin: 30px 0 20px 0; font-size: 1.5em; color: #78350f;">📦 Required Materials</h3>
                    <table class="materials-table">
                        <thead>
                            <tr>
                                <th>Material</th>
                                <th>Quantity Required</th>
                                <th>Estimated Cost</th>
                                <th>Purpose</th>
                            </tr>
                        </thead>
                        <tbody>
                            {materials_html if materials_html else '<tr><td colspan="4" style="text-align: center; padding: 30px;">No materials calculated - No potholes detected</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <div class="section-icon">📋</div>
                    Detailed Detection Results
                </h2>
                <table class="detections-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Severity</th>
                            <th>Confidence</th>
                            <th>Dimensions (W×H)</th>
                            <th>Area (px²)</th>
                            <th>Location (X, Y)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        if detections:
            for idx, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det['bbox']
                severity_class = f"badge-{det['severity']}"
                confidence_percent = det['confidence'] * 100
                
                html_content += f"""
                        <tr>
                            <td><strong>{idx}</strong></td>
                            <td><span class="severity-badge {severity_class}">{det['severity'].upper()}</span></td>
                            <td>
                                <div style="margin-bottom: 8px;">{confidence_percent:.1f}%</div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                                </div>
                            </td>
                            <td>{det['width']} × {det['height']} px</td>
                            <td>{det['area']:,}</td>
                            <td>({x1}, {y1})</td>
                        </tr>
"""
        else:
            html_content += """
                        <tr>
                            <td colspan="6" style="text-align: center; padding: 30px; color: #64748b;">
                                No potholes detected in this image
                            </td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <div class="section-icon">ℹ️</div>
                    Repair Recommendations
                </h2>
                
                <div class="info-box">
                    <div class="info-box-title">🔧 Standard Repair Procedure</div>
                    <div class="info-box-content">
                        <ol style="margin-left: 20px; line-height: 1.8;">
                            <li><strong>Preparation:</strong> Clean and dry the pothole area, remove loose debris</li>
                            <li><strong>Base Layer:</strong> Apply and compact aggregate base course</li>
                            <li><strong>Prime Coat:</strong> Apply prime coat for waterproofing</li>
                            <li><strong>Tack Coat:</strong> Apply tack coat for proper bonding</li>
                            <li><strong>Patching:</strong> Fill with cold mix or hot mix asphalt</li>
                            <li><strong>Compaction:</strong> Compact the patch material properly</li>
                            <li><strong>Seal Coat:</strong> Apply seal coat for weather protection</li>
                            <li><strong>Curing:</strong> Allow proper curing time before opening to traffic</li>
                        </ol>
                    </div>
                </div>
                
                <div class="info-box" style="background: #fee2e2; border-left-color: #ef4444;">
                    <div class="info-box-title" style="color: #991b1b;">⚠️ Important Notes</div>
                    <div class="info-box-content" style="color: #7f1d1d;">
                        • Material estimates are based on standard assumptions and may vary based on actual site conditions<br>
                        • Costs are approximate and may vary by region, supplier, and market conditions<br>
                        • High severity potholes may require structural assessment before repair<br>
                        • Weather conditions must be favorable for hot mix asphalt application<br>
                        • Professional inspection recommended for optimal repair strategy
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🛣️ Pothole Detection & Repair Analysis System v2.0</strong></p>
            <p>Powered by YOLOv8 Deep Learning | FastAPI Framework</p>
            <p>Material Estimation Based on ASTM Standards & Industry Best Practices</p>
            <p style="margin-top: 15px; font-size: 0.9em;">© 2025 Road Safety & Infrastructure Analytics</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content


# Initialize detector
detector = PotholeDetector(model) if model else None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Detection & Repair System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 24px;
            padding: 60px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.4);
            max-width: 650px;
            width: 100%;
        }
        h1 {
            font-size: 2.8em;
            color: #1e293b;
            margin-bottom: 15px;
            text-align: center;
            font-weight: 800;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 40px;
            font-size: 1.15em;
            line-height: 1.6;
        }
        .upload-area {
            border: 3px dashed #6366f1;
            border-radius: 20px;
            padding: 60px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        .upload-area:hover {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
            border-color: #8b5cf6;
            transform: translateY(-2px);
        }
        .upload-icon {
            font-size: 5em;
            margin-bottom: 20px;
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 18px 40px;
            font-size: 1.15em;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            font-weight: 700;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(99, 102, 241, 0.3);
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4);
        }
        .btn:disabled {
            background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
            cursor: not-allowed;
            transform: none;
        }
        #fileName {
            margin: 20px 0;
            text-align: center;
            color: #10b981;
            font-weight: 700;
            font-size: 1.1em;
        }
        #loading {
            display: none;
            text-align: center;
            margin: 20px 0;
            color: #6366f1;
            font-weight: 600;
        }
        .spinner {
            border: 4px solid #e2e8f0;
            border-top: 4px solid #6366f1;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .features {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 30px;
        }
        .feature {
            background: #f8fafc;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid #e2e8f0;
        }
        .feature-icon {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .feature-text {
            font-size: 0.9em;
            color: #64748b;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛣️ Pothole Detector</h1>
        <p class="subtitle">AI-Powered Road Analysis with Material Cost Estimation</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📤</div>
            <p style="font-size: 1.3em; color: #1e293b; margin-bottom: 10px; font-weight: 700;">Click to Upload Road Image</p>
            <p style="color: #64748b; font-size: 1.05em;">Supports JPG, PNG, JPEG formats</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*">
        <div id="fileName"></div>
        
        <div id="loading">
            <div class="spinner"></div>
            <p>🔍 Analyzing road conditions & calculating materials...</p>
        </div>
        
        <button class="btn" id="submitBtn" disabled>🚀 Analyze & Generate Report</button>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">🎯</div>
                <div class="feature-text">High Accuracy Detection</div>
            </div>
            <div class="feature">
                <div class="feature-icon">📊</div>
                <div class="feature-text">Detailed Analytics</div>
            </div>
            <div class="feature">
                <div class="feature-icon">🔨</div>
                <div class="feature-text">Material Estimation</div>
            </div>
            <div class="feature">
                <div class="feature-icon">💰</div>
                <div class="feature-text">Cost Analysis</div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileName.textContent = '✅ ' + e.target.files[0].name;
                submitBtn.disabled = false;
            }
        });
        
        submitBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            submitBtn.disabled = true;
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    window.open('/report/' + result.report_id, '_blank');
                } else {
                    alert('❌ Detection failed. Please try again.');
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)):
    """Endpoint to detect potholes in uploaded image"""
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = detector.process_image(file_path)
        
        report_html = ReportGenerator.generate_html_report(result, file.filename)
        
        report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = OUTPUT_DIR / f"report_{report_id}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        
        return JSONResponse({
            "status": "success",
            "report_id": report_id,
            "detections": len(result['detections']),
            "statistics": result['statistics'],
            "material_estimate": result['material_estimate']
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str):
    """Retrieve generated report"""
    report_path = OUTPUT_DIR / f"report_{report_id}.html"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        total_reports = len(list(OUTPUT_DIR.glob("report_*.html")))
        total_images = len(list(UPLOAD_DIR.glob("*.*")))
        
        return JSONResponse({
            "total_reports": total_reports,
            "total_images_processed": total_images,
            "model_loaded": model is not None,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    print("=" * 70)
    print("🛣️  POTHOLE DETECTION & REPAIR ANALYSIS SYSTEM v2.0")
    print("=" * 70)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Loaded: {'✅ Yes' if model else '❌ No'}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n🚀 Starting FastAPI server...")
    print("📍 Access the application at: http://localhost:8000")
    print("📊 API Documentation at: http://localhost:8000/docs")
    print("🔨 Features: Detection + Material Estimation + Cost Analysis")
    print("\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")