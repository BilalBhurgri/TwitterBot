import random
import datetime
import time
import threading
import subprocess
import os
import arxiv
import smtplib
import uuid
import docker 
from flask import Flask, request
from email.message import EmailMessage
from dotenv import load_dotenv
from bot.twitter_bot import TwitterBot
from try_models.simple_update_db import process_paper

# For timezone handling
from zoneinfo import ZoneInfo

load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Set your timezone
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

app = Flask(__name__)

# Global variables
admin = os.getenv("SENDER")
confirmations = []
bots = []

def send_confirmation_emails(bot, index_of_bot, tweet):
    expiry = datetime.datetime.now(LOCAL_TZ) + datetime.timedelta(hours=1)
    token = str(uuid.uuid4()) + str(index_of_bot)
    confirmations[index_of_bot] = {
        "expiry": expiry,
        "confirmed": False,
        "token": token,
        "tweet": tweet
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
    msg["Subject"] = (f"Please confirm the tweet for Persona bot: {bot.account}") 
    msg["From"] = os.getenv('SENDER')
    msg["To"] = admin

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(os.getenv('SENDER'), os.getenv('SENDER_PASSWORD'))
            print(f"Sending email from {os.getenv('SENDER')} to {admin}")
            smtp.send_message(msg)
    except Exception as e:
        print(f"Error sending email to {admin}: {str(e)}")
    print(f"Prepared tweet:\n{tweet}")
    print(f"Sent confirmation emails to admin")

def post_tweet(bot, tweet):
    bot.post_tweet(tweet)

@app.route("/confirm")
def confirm():
    token = request.args.get("token")
    last_char = token[-1]
    if not last_char.isdigit():
        print("Invalid")
        return "Invalid token.", 400
    record = confirmations[int(last_char)]
    print(record)
    if not record:
        print("Invalid record")
        return "Invalid record.", 400
    if datetime.datetime.now(LOCAL_TZ) > record["expiry"]:
        print("Expired token")
        return "Token expired.", 403
    if record["token"] != token:
        print("Invalid\ntoken: ",record["token"], "\nfoken: ", token)
        return "Invalid token.", 400
    if record["confirmed"]:
        print("Already confirmed")
        return "Already confirmed.", 200

    record["confirmed"] = True
    print(f"✅ Admin confirmed. Tweet will be posted.")
    posted = post_tweet(bots[int(last_char)], record["tweet"])
    confirmations[int(last_char)] = {}
    return "Confirmed!", 200

def run_bot(bot, index_of_bot):
    try:
        tweet = bot.run()
        send_confirmation_emails(bot, index_of_bot, tweet)
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

    # Create an arXiv client
    client = arxiv.Client()

    # Open file to write URLs
    with open('try_models/papers.txt', 'w') as f:
        for paper in client.results(search):
            # Get the PDF URL
            pdf_url = paper.pdf_url
            # Write URL to file
            f.write(f"{pdf_url}\n")
            print(f"Added URL: {pdf_url}")
    try:
        # puts it in chroma db
        subprocess.run(['python', 'try_models/simple_update_db.py', '--name', 'db/papers2/chroma.sqlite3', '--input', 'try_models/papers.txt', '--embedding_model', 'all-MiniLM-L6-v2'], 
                    check=True,
                    cwd=project_root)
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

def schedule_daily_tasks(bot, index_of_bot):
    while True:
        now = datetime.datetime.now(LOCAL_TZ)
        tomorrow = now + datetime.timedelta(days=1)

        for hour in [19]:  # 9 AM and 7 PM
            t = get_random_time_between(hour)
            t = datetime.datetime.now(LOCAL_TZ) + datetime.timedelta(seconds=2)
            if t > now:
                print("Scheduling task at:", t)
                schedule_task_at(t, lambda: run_bot(bot, index_of_bot))
            else:
                t = t.replace(day=tomorrow.day, month=tomorrow.month, year=tomorrow.year)
                print("Scheduling task at (tomorrow):", t)
                schedule_task_at(t, lambda: run_bot(bot, index_of_bot))

        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        time.sleep((next_midnight - datetime.datetime.now(LOCAL_TZ)).total_seconds())
        confirmations = {}

def schedule_daily_paper_retrieval(number_of_papers):
    while True:
        now = datetime.datetime.now(LOCAL_TZ)
        tomorrow = now + datetime.timedelta(days=1)

        for hours in [20]:
            # Bot gathers latest papers
            t = datetime.datetime.now(LOCAL_TZ) + datetime.timedelta(seconds=1)
            if t > now:
                print("Scheduling task at:", t)
                # execute task
                schedule_task_at(t, lambda: fetch_recent_papers(number_of_papers))
            else:
                t = t.replace(day=tomorrow.day, month=tomorrow.month, year=tomorrow.year)
                print("Scheduling task at (tomorrow):", t)
                # execute task
                schedule_task_at(t, lambda: fetch_recent_papers(number_of_papers))
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        time.sleep((next_midnight - datetime.datetime.now(LOCAL_TZ)).total_seconds())

if __name__ == "__main__":
    # client = docker.from_env()
    # container = client.containers.run(
    #     "lfoppiano/grobid:0.8.2",
    #     ports={"8070/tcp": 8080, "8071/tcp": 8081},
    #     remove=True,
    #     init=True,
    #     ulimits=[docker.types.Ulimit(name="core", soft=0, hard=0)],
    #     detach=True
    # )
    # for i in range(6):
    #     bots.append(TwitterBot(post = True, account = i))
    #     confirmations.append({})
    # j = 0
    # for bot in bots:
    #     threading.Thread(target=schedule_daily_tasks, args=(bot, j), daemon=True).start()
    #     j+=1
    # This argument is the number of papers retrieved daily
    threading.Thread(target=schedule_daily_paper_retrieval, args=(100,), daemon=True).start()
    app.run(host="0.0.0.0", port=os.getenv('PORT_NO'))
