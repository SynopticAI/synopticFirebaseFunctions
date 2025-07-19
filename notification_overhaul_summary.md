# Notification System Overhaul Summary

## Background
The notification system was experiencing reliability issues where only 2 out of 10 expected notifications were being delivered, despite logs showing "1 successful, 0 failed" for each trigger.

## Root Causes Identified

### 1. Circular Import Issue
**Problem**: `data_processing_utils.py` had a local import inside the `save_inference_output` function:
```python
def save_inference_output(...):
    try:
        from .notification_utils import trigger_action  # ❌ Circular import
```

**Impact**: Created unpredictable module loading behavior that could interrupt notification processing.

**Solution**: Moved the import to module level:
```python
# At top of data_processing_utils.py
from .notification_utils import trigger_action  # ✅ Clean import
```

### 2. Async Execution Issue
**Problem**: Using `asyncio.run()` inside Firebase Functions:
```python
success = asyncio.run(send_notification(trigger))  # ❌ Problematic in Firebase Functions
```

**Impact**: Firebase Functions already run in an async context. Creating new event loops with `asyncio.run()` can cause incomplete execution where the function appears to succeed but notifications don't actually get delivered.

**Solution**: Made the notification system synchronous:
```python
def send_notification(trigger: NotificationTrigger) -> bool:  # ✅ Synchronous
    # ... notification logic
    
success = send_notification(trigger)  # ✅ Direct call
```

## Current Architecture

### File Structure
```
functions/
├── utils/
│   ├── __init__.py                 # Clean imports for main.py
│   ├── notification_utils.py       # All notification logic
│   └── data_processing_utils.py    # Data processing + inference storage
└── main.py                         # Firebase Functions (imports from utils)
```

### Notification Flow
1. **Inference Processing**: `main.py` processes image → calls `save_inference_output()`
2. **Metrics Extraction**: Extract class counts and detection data from inference results
3. **Trigger Evaluation**: `trigger_action()` checks notification settings for each detected class
4. **Notification Sending**: `send_notification()` sends FCM messages to enabled tokens
5. **Logging**: Success/failure logged to Firestore for debugging

### Key Components

#### NotificationTrigger Dataclass
```python
@dataclass
class NotificationTrigger:
    device_id: str
    user_id: str
    class_name: str
    trigger_type: str          # "count" or "location"
    trigger_reason: str        # Human-readable reason
    detection_count: int
    confidence: float
    timestamp: int
    detection_coordinates: List[Tuple[float, float]] = None
```

#### Trigger Types
- **Count Triggers**: Fire when detection count ≥ threshold
- **Location Triggers**: Fire when detections appear in defined regions

## Current Issue: App State Dependency

### Observed Behavior
- ✅ **App Closed**: Notifications arrive reliably
- ❌ **App Open**: Notifications often don't appear, especially after app restart
- ❌ **After App Restart**: Notification delivery becomes very unreliable

### Potential Causes

#### 1. FCM Token Management
- App restart might invalidate or change FCM tokens
- Tokens might not be properly re-registered with the backend
- `enabledFCMTokens` array might contain stale tokens

#### 2. Notification Channel Issues
- Android notification channels might be getting reset
- App might be overriding notification settings on restart
- Background notification permissions might be changing

#### 3. Firebase SDK State
- App restart might cause Firebase SDK initialization issues
- FCM service might not be properly reconnecting
- Token refresh might not be happening correctly

#### 4. App Lifecycle Interference
- Foreground app might be intercepting notifications
- App might be handling notifications differently when active
- Background processing restrictions might apply

## Next Steps for Investigation

### 1. Token Validation
- Add logging to track FCM token changes on app restart
- Verify `enabledFCMTokens` array stays current
- Check if tokens are being properly refreshed

### 2. Notification Delivery Tracking
- Add server-side delivery confirmation tracking
- Log FCM response details for failed deliveries
- Track notification receipt on client side

### 3. App State Monitoring
- Monitor notification behavior across different app states
- Test notification delivery timing relative to app lifecycle events
- Check if foreground/background state affects delivery

### 4. FCM Configuration Review
- Verify notification channel configuration
- Check Android notification permissions and settings
- Review FCM service initialization in the app

## Technical Debt Resolved
- ✅ Eliminated circular imports
- ✅ Removed problematic async patterns in Firebase Functions
- ✅ Centralized notification logic in dedicated modules
- ✅ Improved code organization and maintainability

## Outstanding Issues
- 🔍 App state-dependent notification delivery
- 🔍 Potential FCM token management issues
- 🔍 Memory usage optimization needed (separate issue)

---

*Last Updated: Current refactoring session*
*Status: Backend notification system stable, investigating client-side delivery issues*