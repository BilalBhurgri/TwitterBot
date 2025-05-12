import random
import datetime
import time
import threading
import subprocess
import os
import smtplib
import uuid
from flask import Flask, request
from email.message import EmailMessage
from dotenv import load_dotenv
from bot.bot import main, post_tweet

# For timezone handling
from zoneinfo import ZoneInfo

load_dotenv()

# Set your timezone
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

app = Flask(__name__)

# Global variables
current_tweet = ""
admins = [
    os.getenv("SENDER")
]
confirmations = {}

def send_confirmation_emails(tweet):
    global confirmations, current_tweet
    expiry = datetime.datetime.now(LOCAL_TZ) + datetime.timedelta(hours=1)
    confirmations = {}  # Reset confirmations for new tweet
    current_tweet = tweet
    for email in admins:
        token = str(uuid.uuid4())
        confirmations[token] = {
            "email": email,
            "expiry": expiry,
            "confirmed": False
        }

        url = f"http://{os.getenv('VM_IP')}:{os.getenv('PORT_NO')}/confirm?token={token}"
        body = (
            f"Hello,\n\n"
            f"A tweet has been submitted for review. Please confirm it by clicking the link below within 1 hour:\n\n"
            f"{url}\n\n"
            f"Tweet preview:\n{tweet}\n\n"
            f"Thank you."
        )

        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = "Please confirm the tweet"
        msg["From"] = os.getenv('SENDER')
        msg["To"] = email

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(os.getenv('SENDER'), os.getenv('SENDER_PASSWORD'))
                print(f"Sending email from {os.getenv('SENDER')} to {email}")
                smtp.send_message(msg)
        except Exception as e:
            print(f"Error sending email to {email}: {str(e)}")
    print(f"Sent confirmation emails to {len(admins)} admins")

@app.route("/confirm")
def confirm():
    global confirmations, current_tweet
    token = request.args.get("token")
    record = confirmations.get(token)
    print("Endpoint works")
    if not record:
        print("Invalid or exp")
        return "Invalid or expired token.", 400
    if datetime.datetime.now(LOCAL_TZ) > record["expiry"]:
        print("E")
        return "Token expired.", 403
    if record["confirmed"]:
        print("A")
        return "Already confirmed.", 200

    record["confirmed"] = True
    if all(c["confirmed"] for c in confirmations.values()):
        print("✅ All admins confirmed. Tweet will be posted.")
        post_tweet(current_tweet)
        current_tweet = ""
        confirmations = {}
    return "Confirmed!", 200

def run_bot():
    try:
        tweet = main()
        send_confirmation_emails(tweet)
    except Exception as e:
        print(f"Error running bot: {str(e)}")

def get_random_time_between(start_hour):
    now = datetime.datetime.now(LOCAL_TZ)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return now.replace(hour=start_hour, minute=minute, second=second, microsecond=0)

def schedule_task_at(random_time, task):
    now = datetime.datetime.now(LOCAL_TZ)
    delay = (random_time - now).total_seconds()
    if delay < 0:
        return
    threading.Timer(delay, task).start()

def schedule_daily_tasks():
    while True:
        now = datetime.datetime.now(LOCAL_TZ)
        tomorrow = now + datetime.timedelta(days=1)

        for hour in [9, 19]:  # 9 AM and 7 PM
            t = get_random_time_between(hour)
            if t > now:
                print("Scheduling task at:", t)
                schedule_task_at(t, run_bot)
            else:
                t = t.replace(day=tomorrow.day, month=tomorrow.month, year=tomorrow.year)
                print("Scheduling task at (tomorrow):", t)
                schedule_task_at(t, run_bot)

        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        time.sleep((next_midnight - datetime.datetime.now(LOCAL_TZ)).total_seconds())

if __name__ == "__main__":
    threading.Thread(target=schedule_daily_tasks, daemon=True).start()
    app.run(host="0.0.0.0", port=os.getenv('PORT_NO'))
