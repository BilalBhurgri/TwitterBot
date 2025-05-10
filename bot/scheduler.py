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

current_tweet = ""
admins = [
    os.getenv("SENDER")
]
confirmations = {}

def send_confirmation_emails(tweet):
    expiry = datetime.datetime.now(LOCAL_TZ) + datetime.timedelta(hours=1)
    for email in admins:
        token = str(uuid.uuid4())
        confirmations[token] = {
            "email": email,
            "expiry": expiry,
            "confirmed": False
        }

        url = f"http://{os.getenv('VM_IP')}:{os.getenv('PORT')}/confirm?token={token}"
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
        current_tweet = tweet

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
    token = request.args.get("token")
    record = confirmations.get(token)

    if not record:
        return "Invalid or expired token.", 400
    if datetime.datetime.now(LOCAL_TZ) > record["expiry"]:
        return "Token expired.", 403
    if record["confirmed"]:
        return "Already confirmed.", 200

    record["confirmed"] = True
    print(f"{record['email']} confirmed.")

    if all(c["confirmed"] for c in confirmations.values()):
        print("✅ All admins confirmed. Tweet will be posted.")
        post_tweet(current_tweet)
        current_tweet = ""
        confirmations = {}
        return "All admins confirmed. Tweet action complete!"
    
    return f"Thanks, {record['email']}. Awaiting other confirmations."

def run_bot():
    try:
        tweet = main()
        send_confirmation_emails(tweet)
    except Exception as e:
        print(f"Error running bot: {str(e)}")

def get_random_time_between(start_hour):
    now = datetime.datetime.now(LOCAL_TZ)
    minute = random.randint(30, 35)
    second = random.randint(0, 20)
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

        for hour in [14, 19]:  # 2 PM and 7 PM
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
    app.run(host="0.0.0.0", port=os.getenv('PORT'))
