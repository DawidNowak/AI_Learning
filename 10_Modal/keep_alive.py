import time
import modal
from datetime import datetime

Expert = modal.Cls.lookup("python-expert", "Expert")
expert = Expert()
while True:
    reply = expert.wake_up.remote()
    print(f"{datetime.now()}: {reply}")
    time.sleep(30)