import requests
import psycopg2
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (to be set by user)
GET_URL = os.getenv("GET_URL")
GET_HEADERS = os.getenv("GET_HEADERS")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_DATABASE"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def perform_get_request(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Perform GET request and return the JSON response."""
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def connect_to_database(config: Dict[str, str]):
    """Connect to the PostgreSQL database and return the connection."""
    return psycopg2.connect(**config)

def store_payload(conn, payload: Dict[str, Any]):
    """Store the payload in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO payloads (data) VALUES (%s)",
            (json.dumps(payload),)
        )
    conn.commit()

def send_slack_notification(webhook_url: str, message: str):
    """Send a notification to Slack."""
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()

def main():
    try:
        # Perform GET request
        payload = perform_get_request(GET_URL, GET_HEADERS)
        logger.info("GET request successful")

        # Connect to database
        with connect_to_database(DB_CONFIG) as conn:
            # Store payload in database
            store_payload(conn, payload)
            logger.info("Payload stored in database")

        # Send success notification
        send_slack_notification(SLACK_WEBHOOK_URL, "Operation completed successfully")
        logger.info("Success notification sent")

    except requests.RequestException as e:
        error_message = f"GET request failed: {str(e)}"
        logger.error(error_message)
        send_slack_notification(SLACK_WEBHOOK_URL, error_message)

    except psycopg2.Error as e:
        error_message = f"Database operation failed: {str(e)}"
        logger.error(error_message)
        send_slack_notification(SLACK_WEBHOOK_URL, error_message)

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        send_slack_notification(SLACK_WEBHOOK_URL, error_message)

if __name__ == "__main__":
    main()