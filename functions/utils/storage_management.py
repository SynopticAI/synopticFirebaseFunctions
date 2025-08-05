from firebase_admin import storage
from firebase_functions import logger
from typing import List, Tuple, Optional, Dict, Any
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re


class StorageManager:
    """
    Handles Firebase Storage operations for the rolling window system.
    Optimized for Firebase Functions with batch operations and efficient cleanup.
    """
    
    @staticmethod
    def list_device_images(user_id: str, device_id: str, folder: str = "receiving") -> List[Tuple[str, int]]:
        """
        List all images for a device, sorted by timestamp (oldest first).
        
        Args:
            user_id: User ID
            device_id: Device ID  
            folder: Storage folder name
            
        Returns:
            List of (blob_name, timestamp) tuples sorted by timestamp
        """
        try:
            bucket = storage.bucket()
            prefix = f"users/{user_id}/devices/{device_id}/{folder}/"
            
            logger.info(f"Listing images with prefix: {prefix}")
            
            # List all blobs with the prefix
            blobs = bucket.list_blobs(prefix=prefix)
            
            image_files = []
            valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
            
            for blob in blobs:
                try:
                    # Extract filename from full path
                    filename = blob.name.split('/')[-1]
                    
                    # Check if it's an image file
                    file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                    if file_ext not in valid_extensions:
                        continue
                    
                    # Extract timestamp from filename (format: timestamp.extension)
                    if '.' in filename:
                        timestamp_str = filename.split('.')[0]
                        try:
                            # Handle both millisecond and second timestamps
                            timestamp = int(timestamp_str)
                            
                            # Convert to milliseconds if it looks like seconds
                            if timestamp < 10000000000:  # Less than year 2286 in seconds
                                timestamp *= 1000
                                
                            image_files.append((blob.name, timestamp))
                            
                        except ValueError:
                            logger.info(f"Could not parse timestamp from filename: {filename}")
                            continue
                    
                except Exception as blob_error:
                    logger.info(f"Error processing blob {blob.name}: {str(blob_error)}")
                    continue
            
            # Sort by timestamp (oldest first)
            image_files.sort(key=lambda x: x[1])
            
            logger.info(f"Found {len(image_files)} valid images for device {device_id}")
            return image_files
            
        except Exception as e:
            logger.error(f"Error listing device images for {device_id}: {str(e)}")
            return []
    
    @staticmethod
    def cleanup_old_images(user_id: str, device_id: str, max_images: int = 10, 
                          folder: str = "receiving") -> Dict[str, Any]:
        """
        Remove oldest images to maintain rolling window of max_images.
        Uses batch operations for efficiency.
        
        Args:
            user_id: User ID
            device_id: Device ID
            max_images: Maximum number of images to keep
            folder: Storage folder name
            
        Returns:
            Dict with cleanup results and metadata
        """
        cleanup_start = time.time()
        
        try:
            # Get all images sorted by timestamp
            image_files = StorageManager.list_device_images(user_id, device_id, folder)
            
            current_count = len(image_files)
            
            if current_count <= max_images:
                logger.info(f"No cleanup needed for device {device_id}. Current: {current_count}, Max: {max_images}")
                return {
                    "cleanup_needed": False,
                    "current_count": current_count,
                    "max_images": max_images,
                    "deleted_count": 0,
                    "processing_time_ms": (time.time() - cleanup_start) * 1000
                }
            
            # Calculate how many to delete
            images_to_delete = current_count - max_images
            files_to_delete = image_files[:images_to_delete]
            
            logger.info(f"Cleaning up {images_to_delete} old images for device {device_id}")
            
            # Delete files using batch operations
            bucket = storage.bucket()
            deleted_count = 0
            failed_deletes = []
            
            # Use ThreadPoolExecutor for concurrent deletions (be careful with limits)
            with ThreadPoolExecutor(max_workers=5) as executor:
                delete_futures = []
                
                for blob_name, timestamp in files_to_delete:
                    future = executor.submit(StorageManager._delete_single_blob, bucket, blob_name)
                    delete_futures.append((future, blob_name, timestamp))
                
                # Collect results
                for future, blob_name, timestamp in delete_futures:
                    try:
                        success = future.result(timeout=10)  # 10 second timeout per delete
                        if success:
                            deleted_count += 1
                            logger.info(f"✅ Deleted: {blob_name}")
                        else:
                            failed_deletes.append(blob_name)
                            logger.info(f"❌ Failed to delete: {blob_name}")
                    except Exception as delete_error:
                        failed_deletes.append(blob_name)
                        logger.error(f"❌ Error deleting {blob_name}: {str(delete_error)}")
            
            processing_time = (time.time() - cleanup_start) * 1000
            
            result = {
                "cleanup_needed": True,
                "current_count": current_count,
                "max_images": max_images,
                "deleted_count": deleted_count,
                "failed_deletes": len(failed_deletes),
                "processing_time_ms": processing_time,
                "remaining_count": current_count - deleted_count
            }
            
            if failed_deletes:
                logger.info(f"Failed to delete {len(failed_deletes)} files: {failed_deletes[:3]}...")
                result["failed_files"] = failed_deletes
            
            logger.info(f"✅ Cleanup completed: {deleted_count}/{images_to_delete} deleted in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old images for device {device_id}: {str(e)}")
            return {
                "cleanup_needed": False,
                "error": str(e),
                "processing_time_ms": (time.time() - cleanup_start) * 1000
            }
    
    @staticmethod
    def _delete_single_blob(bucket, blob_name: str) -> bool:
        """
        Delete a single blob with error handling.
        
        Args:
            bucket: Firebase Storage bucket
            blob_name: Full path of blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                return True
            else:
                logger.info(f"Blob {blob_name} does not exist")
                return True  # Consider non-existent blobs as "successfully deleted"
        except Exception as e:
            logger.error(f"Error deleting blob {blob_name}: {str(e)}")
            return False
    
    @staticmethod
    def get_latest_image(user_id: str, device_id: str, folder: str = "receiving") -> Optional[Tuple[str, bytes, int]]:
        """
        Get the most recent image for a device.
        
        Args:
            user_id: User ID
            device_id: Device ID
            folder: Storage folder name
            
        Returns:
            Tuple of (blob_name, image_bytes, timestamp) or None if no images found
        """
        try:
            # Get all images sorted by timestamp
            image_files = StorageManager.list_device_images(user_id, device_id, folder)
            
            if not image_files:
                logger.info(f"No images found for device {device_id}")
                return None
            
            # Get the latest image (last in sorted list)
            latest_blob_name, latest_timestamp = image_files[-1]
            
            # Download the image
            bucket = storage.bucket()
            blob = bucket.blob(latest_blob_name)
            
            if blob.exists():
                logger.info(f"Downloading latest image: {latest_blob_name}")
                image_bytes = blob.download_as_bytes()
                return (latest_blob_name, image_bytes, latest_timestamp)
            else:
                logger.info(f"Latest image blob does not exist: {latest_blob_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest image for device {device_id}: {str(e)}")
            return None
    
    @staticmethod
    def get_previous_image(user_id: str, device_id: str, current_timestamp: int, 
                          folder: str = "receiving") -> Optional[Tuple[str, bytes, int]]:
        """
        Get the previous image before the current timestamp.
        
        Args:
            user_id: User ID
            device_id: Device ID
            current_timestamp: Current image timestamp
            folder: Storage folder name
            
        Returns:
            Tuple of (blob_name, image_bytes, timestamp) or None if no previous image
        """
        try:
            # Get all images sorted by timestamp
            image_files = StorageManager.list_device_images(user_id, device_id, folder)
            
            if len(image_files) < 2:
                logger.info(f"Not enough images for comparison (found {len(image_files)})")
                return None
            
            # Find the most recent image before current_timestamp
            previous_image = None
            for blob_name, timestamp in reversed(image_files):
                if timestamp < current_timestamp:
                    previous_image = (blob_name, timestamp)
                    break
            
            if not previous_image:
                logger.info("No previous image found")
                return None
            
            blob_name, timestamp = previous_image
            
            # Download the previous image
            bucket = storage.bucket()
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                logger.info(f"Downloading previous image: {blob_name}")
                image_bytes = blob.download_as_bytes()
                return (blob_name, image_bytes, timestamp)
            else:
                logger.info(f"Previous image blob does not exist: {blob_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting previous image for device {device_id}: {str(e)}")
            return None
    
    @staticmethod
    def save_processed_image(user_id: str, device_id: str, image_bytes: bytes, 
                           timestamp: str, content_type: str = "image/jpeg",
                           folder: str = "receiving") -> str:
        """
        Save processed image with overlays to Firebase Storage.
        
        Args:
            user_id: User ID
            device_id: Device ID
            image_bytes: Processed image bytes
            timestamp: Timestamp string for filename
            content_type: MIME type
            folder: Storage folder name
            
        Returns:
            Full path of saved image
        """
        try:
            # Determine file extension based on content type
            extension_map = {
                "image/jpeg": "jpg",
                "image/jpg": "jpg", 
                "image/png": "png",
                "image/webp": "webp"
            }
            
            file_extension = extension_map.get(content_type, "jpg")
            filename = f"{timestamp}.{file_extension}"
            file_path = f"users/{user_id}/devices/{device_id}/{folder}/{filename}"
            
            # Upload to Firebase Storage with optimized settings
            bucket = storage.bucket()
            blob = bucket.blob(file_path)
            
            # Set metadata for better performance
            blob.content_type = content_type
            blob.cache_control = "public, max-age=3600"  # 1 hour cache
            
            # Upload with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    blob.upload_from_string(image_bytes, content_type=content_type)
                    break
                except Exception as upload_error:
                    if attempt == max_retries - 1:
                        raise upload_error
                    logger.info(f"Upload attempt {attempt + 1} failed, retrying: {str(upload_error)}")
                    time.sleep(1)  # Brief delay before retry
            
            logger.info(f"✅ Saved processed image: {file_path} ({len(image_bytes)} bytes)")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            raise e
    
    @staticmethod
    def get_storage_stats(user_id: str, device_id: str, folder: str = "receiving") -> Dict[str, Any]:
        """
        Get storage statistics for a device.
        
        Args:
            user_id: User ID
            device_id: Device ID
            folder: Storage folder name
            
        Returns:
            Dict with storage statistics
        """
        try:
            image_files = StorageManager.list_device_images(user_id, device_id, folder)
            
            if not image_files:
                return {
                    "total_images": 0,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                    "time_span_hours": 0
                }
            
            oldest_timestamp = image_files[0][1]
            newest_timestamp = image_files[-1][1]
            time_span_ms = newest_timestamp - oldest_timestamp
            time_span_hours = time_span_ms / (1000 * 60 * 60)
            
            return {
                "total_images": len(image_files),
                "oldest_timestamp": oldest_timestamp,
                "newest_timestamp": newest_timestamp,
                "time_span_hours": round(time_span_hours, 2),
                "average_interval_minutes": round(time_span_hours * 60 / len(image_files), 2) if len(image_files) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def cleanup_orphaned_files(user_id: str, device_id: str, 
                              valid_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp']) -> Dict[str, Any]:
        """
        Clean up files that don't match expected patterns or extensions.
        
        Args:
            user_id: User ID
            device_id: Device ID
            valid_extensions: List of valid file extensions
            
        Returns:
            Dict with cleanup results
        """
        try:
            bucket = storage.bucket()
            prefix = f"users/{user_id}/devices/{device_id}/receiving/"
            
            # List all blobs
            blobs = bucket.list_blobs(prefix=prefix)
            
            orphaned_files = []
            valid_pattern = re.compile(r'^\d+\.(jpg|jpeg|png|webp)$', re.IGNORECASE)
            
            for blob in blobs:
                filename = blob.name.split('/')[-1]
                
                # Check if filename matches expected pattern
                if not valid_pattern.match(filename):
                    orphaned_files.append(blob.name)
            
            # Delete orphaned files
            deleted_count = 0
            for blob_name in orphaned_files:
                try:
                    blob = bucket.blob(blob_name)
                    blob.delete()
                    deleted_count += 1
                    logger.info(f"Deleted orphaned file: {blob_name}")
                except Exception as delete_error:
                    logger.error(f"Error deleting orphaned file {blob_name}: {str(delete_error)}")
            
            return {
                "orphaned_files_found": len(orphaned_files),
                "deleted_count": deleted_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned files: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }