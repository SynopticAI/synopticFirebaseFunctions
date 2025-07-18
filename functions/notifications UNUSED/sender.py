"""
FCM notification sending logic
"""

import time
from typing import List, Dict, Any
from firebase_functions import logger
from firebase_admin import firestore, messaging
import asyncio

from .models import NotificationTrigger

async def send_notification(trigger: NotificationTrigger) -> bool:
    """
    Send FCM notification only to devices that have enabled notifications for this camera.
    Updated to filter by enabledFCMTokens.
    """
    try:
        db = firestore.client()
        
        # Get device data to check which FCM tokens are enabled
        device_doc = db.collection('users').document(trigger.user_id)\
                      .collection('devices').document(trigger.device_id).get()
        
        if not device_doc.exists:
            logger.error(f"Device {trigger.device_id} not found")
            return False
            
        device_data = device_doc.to_dict()
        enabled_fcm_tokens = device_data.get('enabledFCMTokens', [])
        
        # If no tokens are enabled for this device, skip notification
        if not enabled_fcm_tokens:
            logger.info(f"No FCM tokens enabled for device {trigger.device_id}, skipping notification")
            return False
        
        # Get device name for notification
        device_name = device_data.get('name', 'Unknown Device')
        
        # Build notification message
        title = f"Detection Alert: {trigger.class_name}"
        body = f"{trigger.trigger_reason} on {device_name}"
        
        # Create data payload - only use attributes that exist on NotificationTrigger
        data_payload = {
            'deviceId': trigger.device_id,
            'deviceName': device_name,
            'className': trigger.class_name,
            'triggerType': trigger.trigger_type,
            'triggerReason': trigger.trigger_reason,
            'timestamp': str(trigger.timestamp),
            'confidence': str(trigger.confidence),
            'click_action': 'FLUTTER_NOTIFICATION_CLICK'
        }
        
        # Send notifications to enabled tokens only
        successful_sends = 0
        failed_sends = 0
        
        for token in enabled_fcm_tokens:
            try:
                # Create FCM message
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    data=data_payload,
                    token=token,
                    android=messaging.AndroidConfig(
                        priority='high',
                        notification=messaging.AndroidNotification(
                            channel_id='ai_detection_channel',
                            priority='high',
                            default_sound=True,
                            notification_count=1
                        )
                    ),
                    apns=messaging.APNSConfig(
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                alert=messaging.ApsAlert(
                                    title=title,
                                    body=body
                                ),
                                badge=1,
                                sound='default',
                                category='DETECTION_ALERT'
                            )
                        )
                    )
                )
                
                # Send the message
                response = messaging.send(message)
                logger.info(f"Successfully sent notification: {response}")
                successful_sends += 1
                
            except Exception as send_error:
                logger.error(f"Failed to send notification to token {token[:20]}...: {send_error}")
                failed_sends += 1
                
                # If token is invalid, remove it from device's enabled list
                if "not-registered" in str(send_error).lower() or "invalid" in str(send_error).lower():
                    try:
                        updated_tokens = [t for t in enabled_fcm_tokens if t != token]
                        device_doc.reference.update({
                            'enabledFCMTokens': updated_tokens
                        })
                        logger.info(f"Removed invalid token from device {trigger.device_id}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to remove invalid token: {cleanup_error}")
        
        logger.info(f"Notification complete for device {trigger.device_id}: {successful_sends} successful, {failed_sends} failed")
        return successful_sends > 0
        
    except Exception as e:
        logger.error(f"Error in send_notification: {e}")
        return False

async def _cleanup_invalid_tokens(user_id: str, invalid_tokens: List[Dict[str, Any]]) -> None:
    """Remove invalid FCM tokens from user's token array."""
    try:
        db = firestore.client()
        user_doc_ref = db.collection('users').document(user_id)
        
        # Use transaction to safely remove invalid tokens
        def remove_tokens_transaction(transaction, doc_ref):
            doc = transaction.get(doc_ref)
            if doc.exists:
                user_data = doc.to_dict()
                current_tokens = user_data.get('fcmTokens', [])
                
                # Remove invalid tokens by comparing token strings
                invalid_token_strings = {token_obj.get('token') for token_obj in invalid_tokens}
                cleaned_tokens = [
                    token_obj for token_obj in current_tokens 
                    if token_obj.get('token') not in invalid_token_strings
                ]
                
                transaction.update(doc_ref, {
                    'fcmTokens': cleaned_tokens,
                    'fcmTokenUpdated': firestore.SERVER_TIMESTAMP,
                })
                
                logger.info(f"🧹 Cleaned up {len(invalid_tokens)} invalid tokens for user {user_id}")
        
        db.transaction(lambda t: remove_tokens_transaction(t, user_doc_ref))
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up invalid tokens: {str(e)}")

async def _log_notification_attempt(
    trigger: NotificationTrigger, 
    successful_sends: int, 
    total_tokens: int, 
    error: str = None
) -> None:
    """Log notification attempt for debugging and analytics."""
    try:
        db = firestore.client()
        
        # Get device name
        device_doc = db.collection('users').document(trigger.user_id)\
                      .collection('devices').document(trigger.device_id).get()
        device_name = device_doc.to_dict().get('name', 'Unknown Device') if device_doc.exists else 'Unknown Device'
        
        # Log to user's notification history
        notification_log_ref = db.collection('users').document(trigger.user_id)\
            .collection('notification_logs').document()
            
        log_data = {
            'deviceId': trigger.device_id,
            'deviceName': device_name,
            'className': trigger.class_name,
            'triggerType': trigger.trigger_type,
            'triggerReason': trigger.trigger_reason,
            'timestamp': trigger.timestamp,
            'successfulSends': successful_sends,
            'totalTokens': total_tokens,
            'sent': successful_sends > 0,
        }
        
        if error:
            log_data['error'] = error
            
        notification_log_ref.set(log_data)
        
    except Exception as e:
        logger.error(f"❌ Error logging notification attempt: {str(e)}")