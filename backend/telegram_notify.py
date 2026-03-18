"""Telegram notification system for CoastVision lifeguard alerts."""

import os
import json
import logging
import requests
import threading
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    except Exception as e:
        pass  # Silently ignore .env loading errors

logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get("COASTVISION_TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_API_URL = "https://api.telegram.org/bot"

# Log initialization
if TELEGRAM_BOT_TOKEN:
    logger.info(f"✓ Telegram bot token loaded (token starts with: {TELEGRAM_BOT_TOKEN[:10]}...)")
    print(f"[telegram] ✓ Bot token configured")
else:
    logger.warning("❌ Telegram bot token not found in environment variables")
    print(f"[telegram] ❌ Bot token NOT configured")

# Store Telegram user IDs mapped to lifeguard IDs
TELEGRAM_USERS_FILE = Path(__file__).parent / ".." / "data" / "telegram_users.json"


class TelegramNotifier:
    """Manages Telegram notifications for lifeguard alerts."""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.enabled = bool(self.bot_token)
        self.users: Dict[str, Dict] = {}  # lifeguard_id -> {"chat_id": int, "username": str, ...}
        self.failed_chats = set()  # Track failed chat IDs temporarily
        self.last_errors: Dict[str, str] = {}  # lifeguard_id -> last send error
        self.lock = threading.Lock()
        self._load_users()
    
    def _load_users(self):
        """Load Telegram user mappings from file."""
        if TELEGRAM_USERS_FILE.exists():
            try:
                with open(TELEGRAM_USERS_FILE, "r") as f:
                    data = json.load(f)
                    self.users = data.get("users", {})
                    # Backfill pause flag for older registrations.
                    for lg_id, info in self.users.items():
                        if isinstance(info, dict) and "paused" not in info:
                            info["paused"] = False
                    logger.info(f"Loaded {len(self.users)} Telegram users")
            except Exception as e:
                logger.error(f"Error loading Telegram users: {e}")
    
    def _save_users(self):
        """Save Telegram user mappings to file."""
        try:
            TELEGRAM_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TELEGRAM_USERS_FILE, "w") as f:
                json.dump({"users": self.users}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Telegram users: {e}")
    
    def register_user(self, lifeguard_id: str, chat_id: int, username: str = "") -> bool:
        """Register a lifeguard's Telegram chat ID.
        
        Args:
            lifeguard_id: Unique lifeguard identifier
            chat_id: Telegram chat ID (numeric)
            username: Optional Telegram username
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            logger.warning("Telegram not configured (missing COASTVISION_TELEGRAM_BOT_TOKEN)")
            return False
        
        with self.lock:
            self.users[lifeguard_id] = {
                "chat_id": int(chat_id),
                "username": username,
                "registered_at": datetime.now().isoformat(),
                "paused": False,
            }
            self._save_users()
        
        logger.info(f"Registered lifeguard {lifeguard_id} -> Telegram {chat_id}")
        return True
    
    def unregister_user(self, lifeguard_id: str) -> bool:
        """Unregister a lifeguard's Telegram."""
        with self.lock:
            if lifeguard_id in self.users:
                del self.users[lifeguard_id]
                self._save_users()
                return True
        return False
    
    def get_user(self, lifeguard_id: str) -> Optional[Dict]:
        """Get Telegram info for a lifeguard."""
        with self.lock:
            return self.users.get(lifeguard_id, {}).copy()

    def get_last_error(self, lifeguard_id: str) -> str:
        """Return last Telegram send error for a lifeguard, if any."""
        with self.lock:
            return self.last_errors.get(lifeguard_id, "")

    def set_paused(self, lifeguard_id: str, paused: bool) -> bool:
        """Pause/resume Telegram notifications for a lifeguard."""
        with self.lock:
            if lifeguard_id not in self.users:
                return False
            self.users[lifeguard_id]["paused"] = bool(paused)
            self.users[lifeguard_id]["paused_at"] = datetime.now().isoformat() if paused else None
            self._save_users()
            return True
    
    def send_alert(self, lifeguard_id: str, zone: str, detection_type: str, 
                   confidence: float, image_path: str = None) -> bool:
        """Send an alert notification to a lifeguard.
        
        The text is aligned with the main dashboard alerts, e.g.:
        "Drowning detected in Zone 1 (88.5%)".
        
        Args:
            lifeguard_id: Lifeguard identifier
            zone: Zone name (e.g., "Zone 3")
            detection_type: Type of detection (e.g., "Drowning")
            confidence: Confidence (0-1 float or 0-100 percentage)
            image_path: Optional path to alert image
            
        Returns:
            bool: Delivery status
        """
        if not self.enabled:
            logger.warning("Telegram notifications disabled")
            return False
        
        with self.lock:
            user_info = self.users.get(lifeguard_id)
        
        if not user_info:
            logger.warning(f"No Telegram registered for lifeguard {lifeguard_id}")
            return False

        if user_info.get("paused"):
            logger.info(f"[telegram] Notifications paused for {lifeguard_id}; skipping alert")
            return False
        
        chat_id = user_info.get("chat_id")
        
        # Convert confidence to percentage if needed (0-1 -> 0-100)
        conf_percent = confidence * 100 if confidence <= 1 else confidence

        # Normalize label to match dashboard-style wording
        dt = str(detection_type).strip()
        label_lower = dt.lower()
        if "drown" in label_lower:
            main_text = "Drowning detected"
            emoji = "🚨"
        elif "emerg" in label_lower:
            main_text = "Emergency detected"
            emoji = "🚨"
        else:
            main_text = f"{dt} detected" if dt else "Detection event"
            emoji = "⚠️"

        # Final, concise line similar to dashboard messages
        # Example: "🚨 Drowning detected in Zone 1 (88.5%)"
        message = f"{emoji} {main_text} in {zone} ({conf_percent:.1f}%)"
        
        logger.info(f"[telegram] Sending {detection_type} alert to {lifeguard_id} in {zone} ({conf_percent:.1f}%)")
        return self._send_message(chat_id, message, image_path, recipient_key=lifeguard_id)
    
    def send_crowd_alert(self, zone: str, person_count: int, threshold: int) -> bool:
        """Send crowd density alert to all lifeguards.
        
        Args:
            zone: Zone name
            person_count: Current person count
            threshold: Alert threshold
            
        Returns:
            bool: Success (sent to at least one)
        """
        if not self.enabled:
            return False
        
        message = (
            f"👥 **CROWD ALERT**\n\n"
            f"📍 Zone: {zone}\n"
            f"👤 Count: {person_count} (Threshold: {threshold})\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"Monitor this area closely!"
        )
        
        success = False
        with self.lock:
            for lg_id, user_info in self.users.items():
                if user_info.get("paused"):
                    continue
                chat_id = user_info.get("chat_id")
                if chat_id:
                    if self._send_message(chat_id, message, recipient_key=lg_id):
                        success = True
        
        return success
    
    def _send_message(self, chat_id: int, message: str, image_path: str = None, recipient_key: str = "") -> bool:
        """Send a Telegram message.
        
        Args:
            chat_id: Telegram chat ID
            message: Message text (supports Markdown)
            image_path: Optional image file path
            
        Returns:
            bool: Success status
        """
        try:
            url = f"{TELEGRAM_API_URL}{self.bot_token}"
            
            if image_path and Path(image_path).exists():
                # Send with photo
                with open(image_path, "rb") as img:
                    files = {"photo": img}
                    data = {
                        "chat_id": chat_id,
                        "caption": message,
                        "parse_mode": "Markdown",
                    }
                    response = requests.post(
                        f"{url}/sendPhoto",
                        data=data,
                        files=files,
                        timeout=10
                    )
            else:
                # Send text only
                data = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                }
                response = requests.post(
                    f"{url}/sendMessage",
                    json=data,
                    timeout=10
                )
            
            if response.status_code == 200:
                logger.debug(f"Telegram message sent to {chat_id}")
                with self.lock:
                    if recipient_key:
                        self.last_errors.pop(recipient_key, None)
                    self.failed_chats.discard(chat_id)
                return True
            else:
                logger.warning(f"Telegram send failed ({response.status_code}): {response.text}")
                # Mark as failed if it's a permanent error
                if response.status_code in [400, 401, 403, 404]:
                    self.failed_chats.add(chat_id)
                with self.lock:
                    if recipient_key:
                        self.last_errors[recipient_key] = f"Telegram API {response.status_code}: {response.text[:300]}"
                return False
        
        except requests.exceptions.Timeout:
            logger.warning(f"Telegram timeout for chat {chat_id}")
            with self.lock:
                if recipient_key:
                    self.last_errors[recipient_key] = "Telegram timeout"
            return False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            with self.lock:
                if recipient_key:
                    self.last_errors[recipient_key] = str(e)
            return False
    
    def test_connection(self) -> Dict:
        """Test Telegram bot connection.
        
        Returns:
            dict: Status information
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Bot token not configured",
            }
        
        try:
            url = f"{TELEGRAM_API_URL}{self.bot_token}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                bot_info = response.json().get("result", {})
                return {
                    "status": "connected",
                    "bot_name": bot_info.get("first_name"),
                    "bot_username": bot_info.get("username"),
                    "users_registered": len(self.users),
                }
            else:
                return {
                    "status": "error",
                    "message": f"Invalid bot token ({response.status_code})",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }
    
    def test_user(self, lifeguard_id: str) -> Dict:
        """Test sending a message to a specific user.
        
        Returns:
            dict: Status information
        """
        if not self.enabled:
            return {
                "status": "error",
                "message": "❌ Telegram bot not configured. Set COASTVISION_TELEGRAM_BOT_TOKEN environment variable."
            }
        
        message = (
            "✅ **Test Message**\n\n"
            "Your Telegram connection is working properly!\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        
        user_info = self.get_user(lifeguard_id)
        if not user_info:
            return {"status": "error", "message": "❌ User not registered. Please enter your chat ID first."}
        
        chat_id = user_info.get("chat_id")
        if not chat_id:
            return {"status": "error", "message": "❌ No chat ID found for user."}
        
        success = self._send_message(chat_id, message)
        
        if success:
            return {
                "status": "sent",
                "message": "✓ Test message sent successfully!",
                "chat_id": chat_id,
                "lifeguard_id": lifeguard_id,
            }
        else:
            return {
                "status": "failed",
                "message": "❌ Failed to send test message. Please verify your chat ID is correct and you've started a conversation with the bot.",
                "chat_id": chat_id,
                "lifeguard_id": lifeguard_id,
            }


# Global instance
notifier = TelegramNotifier()
