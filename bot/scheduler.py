import random
import datetime
import time
import threading
import subprocess
import os
import arxiv
import smtplib
import uuid
from bot.tweetBot import TweetBot
from flask import Flask, request
from email.message import EmailMessage
from dotenv import load_dotenv
from bot.bot import main, post_tweet
# from try_models.simple_update_db import process_paper

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
bot = TweetBot()

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
        bot.post_tweet(current_tweet)
        current_tweet = ""
        confirmations = {}
    return "Confirmed!", 200

def run_bot():
    try:
        tweet = main()
        send_confirmation_emails(tweet)
    except Exception as e:
        print(f"Error running bot: {str(e)}")

def fetch_recent_papers(max_papers):
    """Fetch recent deep learning papers from arXiv"""
    # Create search query for deep learning papers
    search = arxiv.Search(
        query="cat:cs.LG",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    # Create output directory if it doesn't exist
    os.makedirs('paper_urls', exist_ok=True)
    # Open file to write URLs
    with open('paper_urls/recent_papers.txt', 'w') as f:
        for paper in search.results():
            # Get the PDF URL
            pdf_url = paper.pdf_url
            # Write URL to file
            f.write(f"{pdf_url}\n")
            print(f"Added URL: {pdf_url}")
    try:
        subprocess.run(['python', 'try_models/simple_update_db.py', '--name', 'db/papers2/chroma.sqlite3', '--input', 'paper_urls/recent_papers.txt'], check=True)
        print("Successfully processed papers with simple_update_db.py")
    except subprocess.CalledProcessError as e:
        print(f"Error running simple_update_db.py: {str(e)}")
    print("Fetched recent papers")



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
        
        # Bot gathers latest papers
        t = get_random_time_between(6)
        if t > now:
            print("Scheduling task at:", t)
            # execute task
            schedule_task_at(t, fetch_recent_papers(100))
        else:
            t = t.replace(day=tomorrow.day, month=tomorrow.month, year=tomorrow.year)
            print("Scheduling task at (tomorrow):", t)
            # execute task
            schedule_task_at(t, run_bot)
        # Bot will post a tweet between 9-10 AM and 6-7 PM
        for hour in [9, 18]:
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
