import json
import os
from firebase_functions import logger

class ConfigLoader:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from secrets.json file"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'secrets.json')
            
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            
            logger.info("✅ Configuration loaded successfully")
            
        except FileNotFoundError:
            logger.error("❌ secrets.json not found. Please create it from secrets.template.json")
            self._config = {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in secrets.json: {e}")
            self._config = {}
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            self._config = {}
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service"""
        if not self._config:
            return ""
        
        api_key = self._config.get(service, {}).get('api_key', '')
        
        if not api_key:
            logger.warning(f"⚠️ No API key found for service: {service}")
        
        return api_key

# Global instance
config = ConfigLoader()

# Convenience functions
def get_moondream_api_key():
    return config.get_api_key('moondream')

def get_gemini_api_key():
    return config.get_api_key('gemini')

def get_openai_api_key():
    return config.get_api_key('openai')

def get_anthropic_api_key():
    return config.get_api_key('anthropic')