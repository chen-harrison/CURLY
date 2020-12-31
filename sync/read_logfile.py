import lcm
from inekf import groundtruth_t

def my_handler(channel, data):
    msg = groundtruth_t.decode(data)
    print("Received message on channel \"{}\"".format(channel))
    print("timestamp = {}".format(msg.mocap_timestamp))
    print("labels: {}".format(msg.contact))
    print("")

lc = lcm.LCM()
subscription = lc.subscribe("ground_truth", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass