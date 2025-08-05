import cv2
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
import io
import time
from firebase_functions import logger
from typing import Tuple, Dict, Any


class ImageSimilarityCalculator:
    """
    Handles various image similarity calculations with configurable algorithms.
    Optimized for Firebase Functions environment with fast histogram comparison as default.
    """
    
    @staticmethod
    def preprocess_image(image_bytes: bytes, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Preprocess image for similarity comparison.
        Resize to standard size and convert to grayscale for consistent comparison.
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize to standard comparison size for faster processing
            image = image.resize(size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and grayscale
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            return gray
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    @staticmethod
    def histogram_similarity(img1_bytes: bytes, img2_bytes: bytes) -> float:
        """
        Compare images using histogram correlation.
        Fast method (~50-100ms), good for lighting changes and general scene similarity.
        Returns similarity score 0.0-1.0 (higher = more similar).
        """
        try:
            img1 = ImageSimilarityCalculator.preprocess_image(img1_bytes)
            img2 = ImageSimilarityCalculator.preprocess_image(img2_bytes)
            
            if img1 is None or img2 is None:
                return 0.0
                
            # Calculate histograms with 64 bins for faster computation
            hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
            
            # Calculate correlation coefficient
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, correlation))
            
        except Exception as e:
            logger.error(f"Error calculating histogram similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def perceptual_hash_similarity(img1_bytes: bytes, img2_bytes: bytes) -> float:
        """
        Compare images using perceptual hashing.
        Medium speed (~100-200ms), good for minor variations, robust to small changes.
        Returns similarity score 0.0-1.0.
        """
        try:
            # Convert to PIL Images
            img1 = Image.open(io.BytesIO(img1_bytes))
            img2 = Image.open(io.BytesIO(img2_bytes))
            
            # Calculate perceptual hashes (8x8 = 64 bit hash)
            hash1 = imagehash.phash(img1, hash_size=8)
            hash2 = imagehash.phash(img2, hash_size=8)
            
            # Calculate Hamming distance
            hamming_distance = hash1 - hash2
            
            # Convert to similarity (lower distance = higher similarity)
            # phash with hash_size=8 uses 64-bit hash, so max distance is 64
            similarity = 1.0 - (hamming_distance / 64.0)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating perceptual hash similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def structural_similarity(img1_bytes: bytes, img2_bytes: bytes) -> float:
        """
        Compare images using Structural Similarity Index (SSIM).
        Slower method (~200-500ms) but most accurate for detecting meaningful changes.
        Returns similarity score 0.0-1.0.
        """
        try:
            img1 = ImageSimilarityCalculator.preprocess_image(img1_bytes, size=(128, 128))
            img2 = ImageSimilarityCalculator.preprocess_image(img2_bytes, size=(128, 128))
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Calculate SSIM with optimized parameters
            similarity_score = ssim(img1, img2, data_range=img1.max() - img1.min())
            
            # SSIM returns -1 to 1, normalize to 0-1
            normalized_score = (similarity_score + 1) / 2
            
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def fast_difference_similarity(img1_bytes: bytes, img2_bytes: bytes) -> float:
        """
        Ultra-fast similarity check using mean squared error.
        Fastest method (~20-50ms), good for detecting major scene changes.
        Returns similarity score 0.0-1.0.
        """
        try:
            # Use smaller size for maximum speed
            img1 = ImageSimilarityCalculator.preprocess_image(img1_bytes, size=(64, 64))
            img2 = ImageSimilarityCalculator.preprocess_image(img2_bytes, size=(64, 64))
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Calculate mean squared error
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            
            # Normalize MSE to similarity score
            # Higher MSE = lower similarity
            max_possible_mse = 255.0 ** 2  # Maximum possible squared difference
            similarity = 1.0 - min(mse / max_possible_mse, 1.0)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating fast difference similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_similarity(img1_bytes: bytes, img2_bytes: bytes, method: str = "histogram") -> Dict[str, Any]:
        """
        Calculate similarity using specified method and return detailed results.
        
        Args:
            img1_bytes: First image as bytes
            img2_bytes: Second image as bytes  
            method: Similarity algorithm ("histogram", "perceptual", "structural", "fast", "combined")
            
        Returns:
            Dict with similarity score and metadata
        """
        start_time = time.time()
        
        try:
            if method == "histogram":
                score = ImageSimilarityCalculator.histogram_similarity(img1_bytes, img2_bytes)
            elif method == "perceptual":
                score = ImageSimilarityCalculator.perceptual_hash_similarity(img1_bytes, img2_bytes)
            elif method == "structural":
                score = ImageSimilarityCalculator.structural_similarity(img1_bytes, img2_bytes)
            elif method == "fast":
                score = ImageSimilarityCalculator.fast_difference_similarity(img1_bytes, img2_bytes)
            elif method == "combined":
                # Use multiple methods and weighted average for best accuracy
                hist_score = ImageSimilarityCalculator.histogram_similarity(img1_bytes, img2_bytes)
                perceptual_score = ImageSimilarityCalculator.perceptual_hash_similarity(img1_bytes, img2_bytes)
                
                # Weighted average: histogram (fast) + perceptual (accurate)
                score = (hist_score * 0.6) + (perceptual_score * 0.4)
            else:
                logger.info(f"Unknown similarity method: {method}, defaulting to histogram")
                score = ImageSimilarityCalculator.histogram_similarity(img1_bytes, img2_bytes)
                method = "histogram"
        
        except Exception as e:
            logger.error(f"Error in similarity calculation: {str(e)}")
            score = 0.0
        
        processing_time = time.time() - start_time
        
        result = {
            "similarity_score": round(score, 4),
            "method": method,
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": int(time.time() * 1000),
            "success": score > 0.0
        }
        
        # Add performance classification
        if processing_time < 0.1:
            result["performance"] = "fast"
        elif processing_time < 0.3:
            result["performance"] = "medium"
        else:
            result["performance"] = "slow"
        
        return result

    @staticmethod
    def get_recommended_threshold(method: str) -> float:
        """
        Get recommended similarity threshold based on the method used.
        
        Args:
            method: Similarity algorithm name
            
        Returns:
            Recommended threshold value (0.0-1.0)
        """
        thresholds = {
            "histogram": 0.85,      # Works well for general scene similarity
            "perceptual": 0.90,     # More strict, good for detecting minor changes
            "structural": 0.80,     # SSIM is more sensitive to changes
            "fast": 0.75,           # MSE-based, needs lower threshold
            "combined": 0.85        # Balanced approach
        }
        
        return thresholds.get(method, 0.85)

    @staticmethod
    def adaptive_threshold(base_threshold: float, recent_scores: list, 
                          target_cache_ratio: float = 0.3) -> float:
        """
        Dynamically adjust similarity threshold based on recent performance.
        
        Args:
            base_threshold: Base threshold value
            recent_scores: List of recent similarity scores
            target_cache_ratio: Target cache hit ratio (0.0-1.0)
            
        Returns:
            Adjusted threshold value
        """
        if not recent_scores or len(recent_scores) < 5:
            return base_threshold
        
        try:
            # Calculate current cache hit ratio with base threshold
            cache_hits = sum(1 for score in recent_scores if score >= base_threshold)
            current_ratio = cache_hits / len(recent_scores)
            
            # Adjust threshold to reach target ratio
            if current_ratio < target_cache_ratio:
                # Too few cache hits, lower threshold
                adjustment = -0.05
            elif current_ratio > target_cache_ratio + 0.1:
                # Too many cache hits, raise threshold  
                adjustment = 0.05
            else:
                # Within acceptable range
                adjustment = 0.0
            
            new_threshold = base_threshold + adjustment
            
            # Keep within reasonable bounds
            new_threshold = max(0.5, min(0.95, new_threshold))
            
            return round(new_threshold, 3)
            
        except Exception as e:
            logger.error(f"Error in adaptive threshold calculation: {str(e)}")
            return base_threshold