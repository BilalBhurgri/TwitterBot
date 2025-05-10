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

load_dotenv()

app = Flask(__name__)

# Global variables, confirm is the number of confirmations needed to post and admins is the list of admins
# The admins have their tokens paired with a random uuid and confirmations is the list of confirmations
current_tweet = ""
admins = [
    os.getenv("SENDER")
    #os.getenv("SENDEE1"),
    #os.getenv("SENDEE2"),
    #os.getenv("SENDEE3")
]
confirmations = {}

# Send confirmation emails to all admins
def send_confirmation_emails(tweet):
    expiry = datetime.datetime.now() + datetime.timedelta(hours=1)
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

        # Send email via Gmail SMTP
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
    if datetime.datetime.now() > record["expiry"]:
        return "Token expired.", 403
    if record["confirmed"]:
        return "Already confirmed.", 200

    record["confirmed"] = True
    print(f"{record['email']} confirmed.")

    if all(c["confirmed"] for c in confirmations.values()):
        print("✅ All admins confirmed. Tweet will be posted.")
        # reset confirmations
        post_tweet(current_tweet)
        current_tweet = ""
        confirmations = {}
        return "All admins confirmed. Tweet action complete!"
    
    return f"Thanks, {record['email']}. Awaiting other confirmations."

def run_bot():
    """Run the bot and handle its output"""
    try:
        # Run the bot script
        tweet = main()
        send_confirmation_emails(tweet)
    except Exception as e:
        print(f"Error running bot: {str(e)}")

def get_random_time_between(start_hour):
    """Return a datetime object with today's date and a random time within the hour."""
    now = datetime.datetime.now()
    minute = random.randint(0, 0)
    second = random.randint(0, 20)
    return now.replace(hour=start_hour, minute=minute, second=second, microsecond=0)

def schedule_task_at(random_time, task):
    delay = (random_time - datetime.datetime.now()).total_seconds()
    if delay < 0:
        return  # Don't schedule if time already passed
    threading.Timer(delay, task).start()

def schedule_daily_tasks():
    while True:
        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(days=1)

        # Schedule for today or tomorrow depending on current time
        for hour in [11, 19]:  # 9AM and 6PM
            t = get_random_time_between(hour)
            if t > now:
                print("Scheduling task at:", t)
                schedule_task_at(t, run_bot)
            else:
                t = t.replace(day=tomorrow.day)
                print("Scheduling task at (tomorrow):", t)
                schedule_task_at(t, run_bot)

        # Sleep until just after midnight to reschedule for the next day
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        print("Sleeping")
        time.sleep((next_midnight - datetime.datetime.now()).total_seconds())

if __name__ == "__main__":
    threading.Thread(target=schedule_daily_tasks, daemon=True).start()
    app.run(host="0.0.0.0", port=os.getenv('PORT'))
