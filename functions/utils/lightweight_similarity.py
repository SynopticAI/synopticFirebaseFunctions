# Create new file: functions/utils/lightweight_similarity.py
# This replaces the heavy image_similarity.py with opencv dependencies

from PIL import Image
import io
import time
import hashlib
from typing import Dict, Any, Tuple
from firebase_functions import logger


class UltraLightSimilarity:
    """
    Ultra-lightweight image similarity using only PIL and built-in Python.
    Optimized for Firebase Functions with minimal dependencies.
    """
    
    @staticmethod
    def average_hash_similarity(img1_bytes: bytes, img2_bytes: bytes, hash_size: int = 8) -> float:
        """
        Compare images using average hash (perceptual hash).
        Very fast and effective for detecting similar images.
        """
        try:
            # Generate hashes for both images
            hash1 = UltraLightSimilarity._average_hash(img1_bytes, hash_size)
            hash2 = UltraLightSimilarity._average_hash(img2_bytes, hash_size)
            
            if not hash1 or not hash2:
                return 0.0
            
            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            
            # Convert to similarity (0-1, where 1 is identical)
            max_distance = len(hash1)
            similarity = 1.0 - (hamming_distance / max_distance)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Average hash similarity failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def _average_hash(image_bytes: bytes, hash_size: int = 8) -> str:
        """Generate average hash for an image using only PIL"""
        try:
            # Open and resize image to small size for speed
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            
            # Get pixel values
            pixels = list(image.getdata())
            
            # Calculate average pixel value
            avg_pixel = sum(pixels) / len(pixels)
            
            # Create hash string: 1 if pixel > average, 0 if <= average
            hash_string = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
            
            return hash_string
            
        except Exception as e:
            logger.error(f"Error generating average hash: {str(e)}")
            return ""
    
    @staticmethod
    def histogram_similarity_lite(img1_bytes: bytes, img2_bytes: bytes) -> float:
        """
        Lightweight histogram comparison using only PIL.
        Faster but less accurate than average hash.
        """
        try:
            # Resize to small size for speed
            img1 = Image.open(io.BytesIO(img1_bytes)).convert('RGB').resize((64, 64))
            img2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB').resize((64, 64))
            
            # Get histograms (PIL returns combined RGB histogram)
            hist1 = img1.histogram()
            hist2 = img2.histogram()
            
            # Simple correlation calculation
            correlation = UltraLightSimilarity._simple_correlation(hist1, hist2)
            return max(0.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Histogram similarity failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def _simple_correlation(hist1: list, hist2: list) -> float:
        """Very simple correlation calculation without numpy"""
        try:
            if len(hist1) != len(hist2):
                return 0.0
            
            # Simple normalized dot product
            dot_product = sum(h1 * h2 for h1, h2 in zip(hist1, hist2))
            norm1 = sum(h1 * h1 for h1 in hist1) ** 0.5
            norm2 = sum(h2 * h2 for h2 in hist2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    @staticmethod
    def calculate_similarity(img1_bytes: bytes, img2_bytes: bytes, method: str = "average_hash") -> Dict[str, Any]:
        """
        Main similarity calculation function.
        
        Args:
            img1_bytes: First image as bytes
            img2_bytes: Second image as bytes
            method: "average_hash" (recommended) or "histogram"
            
        Returns:
            Dict with similarity score and metadata
        """
        start_time = time.time()
        
        try:
            if method == "average_hash":
                similarity_score = UltraLightSimilarity.average_hash_similarity(img1_bytes, img2_bytes)
            elif method == "histogram":
                similarity_score = UltraLightSimilarity.histogram_similarity_lite(img1_bytes, img2_bytes)
            else:
                logger.warning(f"Unknown similarity method: {method}, using average_hash")
                similarity_score = UltraLightSimilarity.average_hash_similarity(img1_bytes, img2_bytes)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "similarity_score": round(similarity_score, 4),
                "method": method,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": int(time.time() * 1000),
                "success": True
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Similarity calculation failed: {str(e)}")
            
            return {
                "similarity_score": 0.0,
                "method": method,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": int(time.time() * 1000),
                "success": False,
                "error": str(e)
            }